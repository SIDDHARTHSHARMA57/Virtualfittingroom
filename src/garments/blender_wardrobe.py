"""
Live wardrobe for the Blender twin: clothing library UI, garment wearing, cloth simulation and
Mirror Mode (garment only over the camera image). Loaded by src/blender_smplx_receiver.py.

Wearing a garment
  1. The processed asset (assets/garments/processed/<id>.blend) is appended. Garments are stored in
     the template body's rest frame.
  2. A hidden DRIVER copy is parented to the twin's rig with a Surface Deform modifier bound to the
     twin at the template shape + the garment's bind pose. From then on the driver follows the
     customer's personalised shape AND live pose: it is where the garment would be if glued on.
  3. The visible DISPLAY copy is moved by the XPBD cloth solver (src/garments/cloth_sim.py) on a
     worker thread: pinned regions follow the driver, the rest hangs, swings and collides with the
     body (own-skin half-space + bone capsules) and the selected person's hands (spheres; pinch = grab).
Mirror Mode
  The twin stays in the scene (it is the collision body and the occluder) but is drawn with the
  live camera image projected through the calibrated camera, so it is visually identical to the
  real person behind it: the garment is hidden exactly where the real arms/hands are in front.
  Rig and other helpers are hidden; only the garment appears on the real camera image.
"""
import json
import math
import os
import subprocess
import sys
import threading
import time

import bpy
import numpy as np
from mathutils import Matrix, Quaternion

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.garments.registry import Registry, load_fabrics  # noqa: E402
from src.garments.cloth_sim import ClothSim  # noqa: E402

KEY = "VFR_WARDROBE"
TWIN, RIG, WORLD = "SMPLX_Twin", "SMPLX_Twin_rig", "VFR_World"
PROCESSOR = os.path.join(ROOT, "src", "garments", "process_garment.py")
BLENDER_TO_SMPL = Matrix.Rotation(-math.pi / 2, 4, "X")   # stored garments are Z-up; the rig frame is Y-up
CAPSULES = [(0, 9), (16, 18), (17, 19), (18, 20), (19, 21), (1, 4), (2, 5), (4, 7), (5, 8), (12, 15)]
HAND = {"left": dict(wrist=20, index1=25, pinky1=31, tips=(27, 30, 33, 36, 39), thumb=39, index=27),
        "right": dict(wrist=21, index1=40, pinky1=46, tips=(42, 45, 48, 51, 54), thumb=54, index=42)}
JOINT_NAMES = None  # filled from the receiver


def _log(msg):
    print(msg)
    try:
        p = os.path.join(ROOT, "outputs", "logs", "blender_twin.log")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    except OSError:
        pass


# ======================================================================================
class Wardrobe:
    def __init__(self, joint_names):
        global JOINT_NAMES
        JOINT_NAMES = joint_names
        self.reg = Registry()
        self.fabrics = load_fabrics()
        self.active = None            # dict for the worn garment
        self.pending = {}             # gid -> {"proc", "report", "started"}
        self.mirror = False
        self.message = ""
        self._radii = None
        self._radii_betas = None
        self._lock = threading.Lock()
        self._job = None
        self._result = None
        self._wake = threading.Event()
        self._alive = True
        self.sim_ms = 0.0
        self.gather_ms = 0.0
        self._pinch = {"left": False, "right": False}
        threading.Thread(target=self._worker, name="VFR-ClothSim", daemon=True).start()

    # ---------------- library ----------------
    def fabric_for(self, g):
        f = dict(self.fabrics["profiles"].get(g.get("fabric_profile"), {}))
        f.update(g.get("fabric_overrides", {}))
        return f

    def ease_for(self, g):
        return self.fabrics["fit_styles"].get(g.get("fit_style", "regular"), 0.012)

    def start_processing(self, path, category="", fit="regular", gid=None, name=""):
        """Copy the upload into the library and process it in a separate Blender process."""
        if gid is None:
            staged = self.reg.stage_upload(path)
            gid, path = staged["id"], staged["original"]
        args = [bpy.app.binary_path, "-b", "--factory-startup", "--python", PROCESSOR, "--",
                "--input", path, "--id", gid, "--library", self.reg.dir, "--fit", fit]
        if category:
            args += ["--category", category]
        if name:
            args += ["--name", name]
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.pending[gid] = {"proc": proc, "report": None, "started": time.time()}
        self.message = f"Processing {os.path.basename(path)}..."
        return gid

    def poll_processing(self):
        for gid, p in self.pending.items():
            if p["report"] is None and p["proc"].poll() is not None:
                rp = os.path.join(self.reg.pending_dir(gid), "report.json")
                p["report"] = json.load(open(rp, encoding="utf-8")) if os.path.isfile(rp) else \
                    {"id": gid, "status": "error", "errors": ["Processor crashed (no report)"], "warnings": [],
                     "name": gid, "category": "?"}
                st = p["report"]["status"]
                self.message = f"{p['report'].get('name', gid)}: " + ("ready for approval" if st == "ok" else "FAILED")

    def approve(self, gid):
        rep = self.pending.pop(gid)["report"]
        entry = self.reg.approve(rep)
        if self.active and self.active["gid"] == gid:  # reprocessed while worn: reload it
            self.wear(gid)
        self.message = f"Added '{entry['name']}' to the library"
        return entry

    def cancel(self, gid):
        self.pending.pop(gid, None)
        self.reg.cancel(gid, delete_original=gid not in self.reg.garments)
        self.message = "Upload cancelled"

    def delete(self, gid, delete_original=False):
        if self.active and self.active["gid"] == gid:
            self.remove()
        removed = self.reg.delete(gid, delete_processed=True, delete_original=delete_original)
        self.message = f"Deleted ({len(removed)} files removed)"

    # ---------------- wearing ----------------
    def wear(self, gid):
        self.remove()
        g = self.reg.garments[gid]
        path = self.reg.abs(g["processed"])
        if not os.path.isfile(path):
            raise RuntimeError(f"Processed asset missing: {path} (reprocess the garment)")
        rig, body = bpy.data.objects.get(RIG), bpy.data.objects.get(TWIN)
        if rig is None or body is None:
            raise RuntimeError("Digital twin not built yet")
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = [n for n in src.objects if n == g["object"]]
        if not dst.objects or dst.objects[0] is None:
            raise RuntimeError(f"Object '{g['object']}' not found in {path}")
        src_ob = dst.objects[0]
        src_ob.data.transform(BLENDER_TO_SMPL)            # into the twin's rig frame (template rest, Y up)
        coll = bpy.data.collections.get("VFR_Garments") or bpy.data.collections.new("VFR_Garments")
        if coll.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(coll)

        driver = src_ob
        driver.name = f"VFR_Driver_{gid}"
        coll.objects.link(driver)
        driver.parent = rig
        driver.matrix_parent_inverse = Matrix.Identity(4)
        driver.location = (0, 0, 0)
        driver.hide_render = True
        driver.display_type = "WIRE"
        driver.hide_set(True)
        mod = driver.modifiers.new("VFR_Follow", "SURFACE_DEFORM")
        mod.target = body
        self._bind(driver, mod, rig, body, float(driver.get("vfr_shoulder_drop_deg", 0.0)))

        display = bpy.data.objects.new(f"VFR_Garment_{gid}", driver.data.copy())
        coll.objects.link(display)
        display.parent = bpy.data.objects[WORLD]
        display.matrix_parent_inverse = Matrix.Identity(4)
        for vg in list(display.vertex_groups):
            display.vertex_groups.remove(vg)

        n = len(driver.data.vertices)
        pin = np.zeros(n)
        vg = driver.vertex_groups.get("vfr_pin")
        if vg is not None:
            for v in driver.data.vertices:
                for e in v.groups:
                    if e.group == vg.index:
                        pin[v.index] = e.weight
        faces = np.array([p.vertices[:] for p in driver.data.polygons], np.int64)
        target = self._driver_target(driver, rig)
        sim = ClothSim(target, faces, pin, self.fabric_for(g), up_axis=1)
        self.active = {"gid": gid, "driver": driver, "display": display, "sim": sim, "faces": faces,
                       "ease": float(driver.get("vfr_ease", self.ease_for(g))), "t_last": time.perf_counter(),
                       "betas": list(rig.get("vfr_betas", [0.0] * 10))}
        self._apply_positions(target)
        if self.mirror:
            self.set_mirror(True)
        self.message = f"Wearing '{g['name']}'"
        _log(f"[VFR] Wearing {g['name']} ({gid}): {n} vertices")

    def _bind(self, driver, mod, rig, body, shoulder_drop_deg):
        """Bind against the TEMPLATE body in the garment's bind pose (shape key and pose correctives
        muted, bones at rest except the shoulders), then restore the live state."""
        keys = body.data.shape_keys.key_blocks
        saved_vals = {k.name: k.value for k in keys}
        saved_rot = {pb.name: pb.rotation_quaternion.copy() for pb in rig.pose.bones}
        try:
            for k in keys:
                if k.name != "Basis":
                    k.value = 0.0
            for pb in rig.pose.bones:
                pb.rotation_quaternion = Quaternion()
            a = math.radians(shoulder_drop_deg)
            rig.pose.bones["left_shoulder"].rotation_quaternion = Quaternion((0, 0, 1), -a)
            rig.pose.bones["right_shoulder"].rotation_quaternion = Quaternion((0, 0, 1), a)
            bpy.context.view_layer.update()
            with bpy.context.temp_override(object=driver, active_object=driver):
                bpy.ops.object.surfacedeform_bind(modifier=mod.name)
            if not mod.is_bound:
                raise RuntimeError("Surface Deform could not bind the garment to the body")
        finally:
            for k in keys:
                k.value = saved_vals[k.name]
            for name, q in saved_rot.items():
                rig.pose.bones[name].rotation_quaternion = q
            bpy.context.view_layer.update()

    def remove(self):
        a, self.active = self.active, None
        with self._lock:
            self._job = self._result = None
        if a is None:
            return
        for ob in (a["display"], a["driver"]):
            me = ob.data
            bpy.data.objects.remove(ob, do_unlink=True)
            if me.users == 0:
                bpy.data.meshes.remove(me)
        self.message = "Garment removed"

    # ---------------- per tick ----------------
    def _driver_target(self, driver, rig):
        ev = driver.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        return co.reshape(-1, 3) + np.array(rig.location)   # rig frame -> VFR_World frame

    def _body_colliders(self, rig, body):
        loc = np.array(rig.location)
        heads = {i: np.array(rig.pose.bones[JOINT_NAMES[i]].head) + loc for i in range(55)}
        betas = list(rig.get("vfr_betas", [0.0] * 10))
        if self._radii is None or betas != self._radii_betas:
            self._radii = self._measure_radii(body, rig)
            self._radii_betas = betas
        A = [heads[a] for a, _ in CAPSULES]
        B = [heads[b] for _, b in CAPSULES]
        R = [self._radii[(a, b)] for a, b in CAPSULES]
        # torso is wider than deep: two side-by-side capsules along the body's left axis
        r_t, off = self._radii["torso2"]
        for j0, j1 in ((0, 9),):
            ax0 = np.array(rig.pose.bones[JOINT_NAMES[j0]].matrix.col[0][:3])
            ax1 = np.array(rig.pose.bones[JOINT_NAMES[j1]].matrix.col[0][:3])
            for s in (1, -1):
                A.append(heads[j0] + s * off * ax0)
                B.append(heads[j1] + s * off * ax1)
                R.append(r_t)
        A, B, R = np.array(A), np.array(B), np.array(R)
        hands = {}
        for side, h in HAND.items():
            palm = (heads[h["wrist"]] + heads[h["index1"]] + heads[h["pinky1"]]) / 3
            tips = [heads[t] for t in h["tips"]]
            hands[side] = (np.array([palm] + tips), np.array([0.035] + [0.011] * len(tips)))
        return (A, B, R), hands, heads

    def _measure_radii(self, body, rig):
        """Capsule radii from the current (personalised) rest mesh: 0.85 x median skin distance."""
        keys = body.data.shape_keys.key_blocks
        co = np.empty(len(body.data.vertices) * 3)
        keys["VFR_BodyShape"].data.foreach_get("co", co)
        V = co.reshape(-1, 3)
        # by name: rig.data.bones is NOT in creation order, so positional indexing mixes up joints
        J = np.array([np.array(rig.data.bones[n].head_local) for n in JOINT_NAMES])
        owner = self._owner(body)
        radii = {}
        for a, b in CAPSULES:
            P = V[owner == a]
            ab = J[b] - J[a]
            t = np.clip(((P - J[a]) @ ab) / max(ab @ ab, 1e-9), 0, 1)
            d = np.linalg.norm(P - (J[a] + t[:, None] * ab), axis=1)
            radii[(a, b)] = 0.85 * float(np.median(d)) if len(d) else 0.05
        T = V[np.isin(owner, (0, 3, 6, 9))]
        half_depth = 0.5 * (np.percentile(T[:, 2], 90) - np.percentile(T[:, 2], 10))
        half_width = 0.5 * (np.percentile(T[:, 0], 90) - np.percentile(T[:, 0], 10))
        radii["torso2"] = (0.9 * half_depth, max(0.0, half_width - half_depth))
        return radii

    def _owner(self, body):
        if getattr(self, "_owner_cache", None) is None:
            W = np.zeros((len(body.data.vertices), len(body.vertex_groups)))
            for v in body.data.vertices:
                for e in v.groups:
                    W[v.index, e.group] = e.weight
            names = [vg.name for vg in body.vertex_groups]
            self._owner_cache = np.array([JOINT_NAMES.index(names[i]) for i in W.argmax(1)])
        return self._owner_cache

    def tick(self):
        """Main thread: gather inputs for the solver, apply its latest result. Never blocks on it."""
        self.poll_processing()
        a = self.active
        if a is None:
            return False
        rig, body = bpy.data.objects.get(RIG), bpy.data.objects.get(TWIN)
        if rig is None or body is None:
            return False
        t0 = time.perf_counter()
        target = self._driver_target(a["driver"], rig)
        caps, hands, heads = self._body_colliders(rig, body)
        betas = list(rig.get("vfr_betas", [0.0] * 10))
        reshape = betas != a["betas"]
        a["betas"] = betas
        grabs = {}
        for side, h in HAND.items():  # pinch (thumb tip ~ index tip) = grab, open = release
            mid = (heads[h["thumb"]] + heads[h["index"]]) / 2
            d = np.linalg.norm(heads[h["thumb"]] - heads[h["index"]])
            if not self._pinch[side] and d < 0.03:
                self._pinch[side] = True
                grabs[side] = ("grab", mid)
            elif self._pinch[side] and d > 0.045:
                self._pinch[side] = False
                grabs[side] = ("release", None)
            if self._pinch[side]:  # a pinching hand holds the fabric at the pinch point, it does not push
                hands[side] = (mid[None], np.zeros(1))
        now = time.perf_counter()
        dt, a["t_last"] = now - a["t_last"], now
        self.gather_ms = (time.perf_counter() - t0) * 1000
        with self._lock:
            self._job = (a["sim"], dt, target, caps, hands, grabs, a["ease"], reshape)
        self._wake.set()
        with self._lock:
            res, self._result = self._result, None
        if res is not None and self.active is a:
            self._apply_positions(res)
            return True
        return False

    def _worker(self):
        while self._alive:
            self._wake.wait(0.5)
            self._wake.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            sim, dt, target, caps, hands, grabs, ease, reshape = job
            t0 = time.perf_counter()
            if reshape:
                sim.set_rest(target)           # new body shape: the fitted garment's lengths change
            for side, (what, where) in grabs.items():
                if what == "grab":
                    sim.grab(side, where, 0.05)
                else:
                    sim.release(side)
            x = sim.step(dt, target, caps, hands, ease)
            self.sim_ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self._result = x.copy()

    def _apply_positions(self, x):
        me = self.active["display"].data
        me.vertices.foreach_set("co", x.astype(np.float32).ravel())
        me.update()

    # ---------------- mirror mode ----------------
    def set_mirror(self, on: bool):
        """Garment-only view: twin drawn with the camera image projected onto it (invisible but
        occluding), rig hidden, camera background shown as an unlit plane (works in Material Preview
        and renders), colour management neutral so the image is not tone-mapped."""
        self.mirror = on
        body, rig = bpy.data.objects.get(TWIN), bpy.data.objects.get(RIG)
        plane, cam = bpy.data.objects.get("VFR_Background"), bpy.data.objects.get("VFR_Camera")
        if body is None:
            return
        if on:
            mat = projected_feed_material(cam)
            if body.data.materials:
                body.data.materials[0] = mat
            else:
                body.data.materials.append(mat)
            if plane is not None:
                plane.hide_viewport = False
            rig.hide_set(True)
            bpy.context.scene.view_settings.view_transform = "Standard"
            for area in bpy.context.screen.areas if bpy.context.screen else []:
                if area.type == "VIEW_3D":
                    area.spaces.active.shading.type = "MATERIAL"
                    area.spaces.active.overlay.show_overlays = False
        else:
            body.data.materials.clear()
            if plane is not None:
                plane.hide_viewport = True
            rig.hide_set(False)
            for area in bpy.context.screen.areas if bpy.context.screen else []:
                if area.type == "VIEW_3D":
                    area.spaces.active.shading.type = "SOLID"
                    area.spaces.active.overlay.show_overlays = True
        self.message = "Mirror Mode on" if on else "Mirror Mode off (debug view)"


def projected_feed_material(cam):
    """Emission material showing VFR_CameraFeed projected through the camera intrinsics:
    U = fx/w * X/Z + cx/w,  V = 1 - cy/h + fy/h * Y/Z   (Blender 'Camera' texture coordinates)."""
    mat = bpy.data.materials.get("VFR_ProjectedFeed") or bpy.data.materials.new("VFR_ProjectedFeed")
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    K = json.loads(cam["vfr_K"]) if cam is not None and "vfr_K" in cam else \
        {"w": 640, "h": 480, "fx": 554.3, "fy": 554.3, "cx": 320, "cy": 240}
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Camera"], sep.inputs[0])

    def math_node(op, a, b=None, val=None):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = op
        nt.links.new(a, n.inputs[0])
        if b is not None:
            nt.links.new(b, n.inputs[1])
        if val is not None:
            n.inputs[1].default_value = val
        return n.outputs[0]

    # Blender's "Camera" texture coordinate has +Z pointing INTO the scene (measured), unlike view
    # space; X right, Y up. So u = fx * X / Z + cx.
    z = sep.outputs["Z"]
    xr, yr = math_node("DIVIDE", sep.outputs["X"], z), math_node("DIVIDE", sep.outputs["Y"], z)
    u = math_node("ADD", math_node("MULTIPLY", xr, val=K["fx"] / K["w"]), val=K["cx"] / K["w"])
    v = math_node("ADD", math_node("MULTIPLY", yr, val=K["fy"] / K["h"]), val=1.0 - K["cy"] / K["h"])
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(u, comb.inputs[0])
    nt.links.new(v, comb.inputs[1])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.get("VFR_CameraFeed")
    tex.extension = "CLIP"
    tex.interpolation = "Closest"   # same sampling as the background plane -> seamless, truly invisible
    nt.links.new(comb.outputs[0], tex.inputs["Vector"])
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return mat


# ======================================================================================
# UI (3D View sidebar > VTryOn)
def W() -> Wardrobe:
    return bpy.app.driver_namespace.get(KEY)


def _categories(self, context):
    items = [("all", "All", "")]
    w = W()
    if w:
        for r, cats in w.fabrics["regions"].items():
            items.append((r, r.capitalize(), f"{r} body"))
    return items


def _garment_items(self, context):
    w = W()
    if not w:
        return [("", "(no library)", "")]
    gs = w.reg.list(context.window_manager.vfr_category)
    return [(g["id"], g["name"], f"{g['category']} ({g['id']})") for g in gs] or [("", "(empty)", "")]


_previews = None


def _preview_icon(g):
    global _previews
    import bpy.utils.previews
    if _previews is None:
        _previews = bpy.utils.previews.new()
    p = W().reg.abs(g["preview"]) if g.get("preview") else ""
    if not p or not os.path.isfile(p):
        return 0
    key = f"{g['id']}:{os.path.getmtime(p)}"
    if key not in _previews:
        _previews.load(key, p, "IMAGE")
    return _previews[key].icon_id


class VFR_OT_wear(bpy.types.Operator):
    bl_idname = "vfr.wear"
    bl_label = "Wear"
    bl_description = "Put the selected garment on the customer"

    def execute(self, context):
        gid = context.window_manager.vfr_garment
        try:
            W().wear(gid)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class VFR_OT_remove(bpy.types.Operator):
    bl_idname = "vfr.remove"
    bl_label = "Remove Garment"

    def execute(self, context):
        W().remove()
        return {"FINISHED"}


class VFR_OT_mirror(bpy.types.Operator):
    bl_idname = "vfr.mirror"
    bl_label = "Toggle Mirror Mode"
    bl_description = "Garment only over the live camera image (twin invisible) / debug view"

    def execute(self, context):
        W().set_mirror(not W().mirror)
        return {"FINISHED"}


class VFR_OT_upload(bpy.types.Operator):
    bl_idname = "vfr.upload"
    bl_label = "Upload Clothing"
    bl_description = "Import a 3D garment (FBX/OBJ/GLB/glTF/.blend) and prepare it for the library"
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.fbx;*.obj;*.glb;*.gltf;*.blend;*.png;*.jpg;*.jpeg",
                                          options={"HIDDEN"})
    category: bpy.props.EnumProperty(name="Category", items=[("", "Detect automatically", "")] + [
        (c, c, "") for c in ("tshirt", "polo", "shirt", "hoodie", "sweatshirt", "sweater", "jacket", "jeans",
                             "trousers", "shorts", "skirt", "dress", "jumpsuit")])
    fit: bpy.props.EnumProperty(name="Fit", items=[(f, f, "") for f in ("tight", "regular", "loose", "oversized")],
                                default="regular")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            W().start_processing(self.filepath, self.category, self.fit)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class VFR_OT_approve(bpy.types.Operator):
    bl_idname = "vfr.approve"
    bl_label = "Add to Clothing Library"
    gid: bpy.props.StringProperty()

    def execute(self, context):
        W().approve(self.gid)
        return {"FINISHED"}


class VFR_OT_cancel_upload(bpy.types.Operator):
    bl_idname = "vfr.cancel_upload"
    bl_label = "Cancel Upload"
    gid: bpy.props.StringProperty()

    def execute(self, context):
        W().cancel(self.gid)
        return {"FINISHED"}


class VFR_OT_delete(bpy.types.Operator):
    bl_idname = "vfr.delete"
    bl_label = "Delete Garment"
    bl_description = "Remove the selected garment from the library"
    confirm: bpy.props.BoolProperty(name="Yes, delete it from the library and its processed files")
    delete_original: bpy.props.BoolProperty(name="Also delete the original uploaded file", default=False)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        g = W().reg.garments.get(context.window_manager.vfr_garment, {})
        self.layout.label(text=f"Delete '{g.get('name', '?')}'?", icon="ERROR")
        self.layout.prop(self, "confirm")
        self.layout.prop(self, "delete_original")

    def execute(self, context):
        if not self.confirm:
            self.report({"WARNING"}, "Not deleted (confirmation not ticked)")
            return {"CANCELLED"}
        W().delete(context.window_manager.vfr_garment, self.delete_original)
        return {"FINISHED"}


class VFR_OT_edit(bpy.types.Operator):
    bl_idname = "vfr.edit"
    bl_label = "Edit Garment"
    name: bpy.props.StringProperty(name="Name")
    category: bpy.props.StringProperty(name="Category")
    fit_style: bpy.props.EnumProperty(name="Fit", items=[(f, f, "") for f in ("tight", "regular", "loose", "oversized")])
    fabric_profile: bpy.props.StringProperty(name="Fabric profile")
    mass: bpy.props.FloatProperty(name="Mass (kg/m2)", min=0.01, max=2.0)
    stretch: bpy.props.FloatProperty(name="Stretch compliance", min=0.0, max=1e-3, precision=8)
    bend: bpy.props.FloatProperty(name="Bend compliance", min=0.0, max=1.0, precision=5)
    damping: bpy.props.FloatProperty(name="Damping (1/s)", min=0.0, max=20.0)
    friction: bpy.props.FloatProperty(name="Friction", min=0.0, max=1.0)

    def invoke(self, context, event):
        g = W().reg.garments[context.window_manager.vfr_garment]
        f = W().fabric_for(g)
        self.name, self.category, self.fit_style = g["name"], g["category"], g.get("fit_style", "regular")
        self.fabric_profile = g["fabric_profile"]
        self.mass, self.stretch, self.bend = f.get("mass", 0.15), f.get("stretch_compliance", 1e-6), f.get("bend_compliance", 2e-3)
        self.damping, self.friction = f.get("damping", 1.0), f.get("friction", 0.3)
        return context.window_manager.invoke_props_dialog(self, width=420)

    def execute(self, context):
        w, gid = W(), context.window_manager.vfr_garment
        w.reg.update(gid, name=self.name, category=self.category, fit_style=self.fit_style,
                     fabric_profile=self.fabric_profile,
                     fabric_overrides={"mass": self.mass, "stretch_compliance": self.stretch,
                                       "bend_compliance": self.bend, "damping": self.damping, "friction": self.friction})
        if w.active and w.active["gid"] == gid:  # live update of the worn garment
            w.active["sim"].f.update(w.fabric_for(w.reg.garments[gid]))
        return {"FINISHED"}


class VFR_OT_reprocess(bpy.types.Operator):
    bl_idname = "vfr.reprocess"
    bl_label = "Reprocess"
    bl_description = "Fit the garment again from its original upload (category / fit style as edited)"

    def execute(self, context):
        w, gid = W(), context.window_manager.vfr_garment
        g = w.reg.garments[gid]
        w.start_processing(w.reg.abs(g["original"]), g["category"], g.get("fit_style", "regular"), gid=gid, name=g["name"])
        return {"FINISHED"}


class VFR_OT_refresh(bpy.types.Operator):
    bl_idname = "vfr.refresh"
    bl_label = "Refresh Library"

    def execute(self, context):
        W().reg.load()
        return {"FINISHED"}


class VFR_PT_wardrobe(bpy.types.Panel):
    bl_label = "VTryOn Clothing Library"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VTryOn"

    def draw(self, context):
        w, wm, L = W(), context.window_manager, self.layout
        if w is None:
            L.label(text="Twin not started")
            return
        row = L.row()
        row.scale_y = 1.6
        row.operator("vfr.mirror", text="Mirror Mode: ON" if w.mirror else "Mirror Mode: OFF",
                     icon="HIDE_ON" if w.mirror else "HIDE_OFF", depress=w.mirror)
        L.prop(wm, "vfr_category", text="Category")
        L.prop(wm, "vfr_garment", text="")
        g = w.reg.garments.get(wm.vfr_garment)
        if g:
            icon = _preview_icon(g)
            if icon:
                L.template_icon(icon_value=icon, scale=7)
            L.label(text=f"{g['name']}  -  {g['category']} ({g['region']})")
            L.label(text=f"Fit: {g.get('fit_style')}  Fabric: {g['fabric_profile']}")
            row = L.row()
            row.scale_y = 1.6
            row.operator("vfr.wear", icon="MOD_CLOTH")
            row.operator("vfr.remove", text="Remove", icon="X")
            row = L.row()
            row.operator("vfr.edit", icon="PREFERENCES")
            row.operator("vfr.reprocess", icon="FILE_REFRESH")
            row.operator("vfr.delete", icon="TRASH")
        row = L.row()
        row.operator("vfr.upload", icon="IMPORT")
        row.operator("vfr.refresh", icon="FILE_REFRESH")
        for gid, p in list(w.pending.items()):
            box = L.box()
            rep = p["report"]
            if rep is None:
                box.label(text=f"Processing {gid}... {time.time() - p['started']:.0f} s", icon="TIME")
                continue
            box.label(text=f"{rep.get('name', gid)} - {rep.get('category', '?')}",
                      icon="CHECKMARK" if rep["status"] == "ok" else "ERROR")
            for e in rep.get("errors", [])[:4]:
                box.label(text=e[:90], icon="ERROR")
            for wv in rep.get("warnings", [])[:4]:
                box.label(text=wv[:90], icon="INFO")
            st = rep.get("stats", {})
            if st:
                box.label(text=f"{st.get('vertices', '?')} verts, moved {st.get('moved_to_fit_pct', '?')}%, "
                               f"inside {st.get('inside_body_pct', '?')}%")
            row = box.row()
            if rep["status"] == "ok":
                row.operator("vfr.approve", icon="CHECKMARK").gid = gid
            row.operator("vfr.cancel_upload", icon="CANCEL").gid = gid
        if w.message:
            L.label(text=w.message, icon="INFO")
        if w.active:
            L.label(text=f"cloth {w.sim_ms:.0f} ms/step, gather {w.gather_ms:.1f} ms")


CLASSES = (VFR_OT_wear, VFR_OT_remove, VFR_OT_mirror, VFR_OT_upload, VFR_OT_approve, VFR_OT_cancel_upload,
           VFR_OT_delete, VFR_OT_edit, VFR_OT_reprocess, VFR_OT_refresh, VFR_PT_wardrobe)


def register(joint_names):
    if bpy.app.driver_namespace.get(KEY) is None:
        bpy.app.driver_namespace[KEY] = Wardrobe(joint_names)
    for c in CLASSES:
        try:
            bpy.utils.register_class(c)
        except ValueError:  # already registered (script re-run)
            bpy.utils.unregister_class(c)
            bpy.utils.register_class(c)
    bpy.types.WindowManager.vfr_category = bpy.props.EnumProperty(name="Category", items=_categories)
    bpy.types.WindowManager.vfr_garment = bpy.props.EnumProperty(name="Garment", items=_garment_items)
    return bpy.app.driver_namespace[KEY]
