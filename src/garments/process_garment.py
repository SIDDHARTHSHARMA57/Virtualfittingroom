"""
Headless garment processor (runs in its own Blender process so uploads never stall the mirror).

  blender -b --factory-startup --python src/garments/process_garment.py -- --input FILE --id GID
          [--name NAME] [--category CAT] [--fit regular] [--library DIR] [--rotate180]
  blender -b --factory-startup --python src/garments/process_garment.py -- --generate tshirt|trousers --out F.obj

Pipeline (writes <library>/pending/<id>/report.json, garment.blend, preview.png):
  import (OBJ/FBX/glTF/GLB/.blend) -> canonical triangle mesh (evaluated, world space, UVs per corner,
  vertices welded across UV seams so the cloth cannot tear there) -> units/orientation fix -> category
  -> placement on the template SMPL-X body -> arm angle of the garment (A-pose vs T-pose) -> body posed
  to match -> conform: only vertices inside the body or closer than the fit ease are pushed out, then
  smoothed (no uniform scaling; loose design parts are kept) -> pin weights by body region ->
  validation -> fabric profile -> preview -> report.
Garments are stored in the SMPL-X rest frame of the template body (Y up, +Z forward, metres); the
live system binds them to the customer's personalised, posed body with Surface Deform.
Photos are rejected: a photo has no 3D garment geometry and needs a separate reconstruction step.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.garments.registry import Registry, load_fabrics, IMAGE_FORMATS  # noqa: E402

NPZ = os.path.join(ROOT, "models", "smplx", "SMPLX_NEUTRAL.npz")
MAX_VERTS = 6000            # cloth solver budget (vertices) for real-time use
KEYWORDS = [("jumpsuit", "jumpsuit"), ("dress", "dress"), ("hoodie", "hoodie"), ("sweatshirt", "sweatshirt"),
            ("sweater", "sweater"), ("jacket", "jacket"), ("polo", "polo"), ("t-shirt", "tshirt"),
            ("tshirt", "tshirt"), ("tee", "tshirt"), ("shirt", "shirt"), ("jean", "jeans"),
            ("trouser", "trousers"), ("pant", "trousers"), ("short", "shorts"), ("skirt", "skirt")]


# ---------------- template body (SMPL-X rest frame) ----------------
class Body:
    def __init__(self):
        d = np.load(NPZ, allow_pickle=True)
        self.v = d["v_template"].astype(np.float64)
        self.f = d["f"].astype(np.int64)
        self.W = d["weights"].astype(np.float64)
        self.owner = self.W.argmax(1)
        self.J = d["J_regressor"].astype(np.float64) @ self.v
        self.parents = d["kintree_table"][0].astype(np.int64)
        self.vt, self.ft = d["vt"].astype(np.float64), d["ft"].astype(np.int64)
        self.posed = self.v

    def pose_arms(self, angle_deg: float):
        """Lower both arms by angle_deg about the shoulder joints (LBS with the real skin weights)."""
        if abs(angle_deg) < 1e-3:
            self.posed = self.v
            return
        desc = {16: [], 17: []}
        for j in range(len(self.parents)):
            k = j
            while k > 0:
                if k in desc:
                    desc[k].append(j)
                    break
                k = self.parents[k]
        out = self.v.copy()
        for sj, sign in ((16, -1.0), (17, 1.0)):
            a = math.radians(angle_deg) * sign
            R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            w = self.W[:, desc[sj]].sum(1)[:, None]
            c = self.J[sj]
            out += w * (((self.v - c) @ R.T + c) - self.v)
        self.posed = out

    def normals(self, v=None):
        v = self.posed if v is None else v
        fn = np.cross(v[self.f[:, 1]] - v[self.f[:, 0]], v[self.f[:, 2]] - v[self.f[:, 0]])
        n = np.zeros_like(v)
        for k in range(3):
            np.add.at(n, self.f[:, k], fn)
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)

    def bvh(self):
        return BVHTree.FromPolygons([tuple(p) for p in self.posed], [tuple(t) for t in self.f])

    @property
    def anchors(self):
        J = self.J
        return {"neck_y": J[12, 1], "shoulder_y": J[[16, 17], 1].mean(), "shoulder_x": abs(J[16, 0]),
                "waist_y": J[3, 1], "hips_y": J[[1, 2], 1].mean(), "chest_z": J[9, 2], "pelvis_z": J[0, 2],
                "ankle_y": J[[7, 8], 1].mean()}


# ---------------- test garments generated from the body ----------------
def a_neck(b: "Body") -> np.ndarray:
    """Neck joint position (centre of the neckline cut-out)."""
    return b.J[12]


def generate(kind: str, out_path: str, ease: float = 0.012):
    """A simulation-ready test garment authored on the template body (like a CLO3D garment made
    on the base avatar): a body region offset outward by `ease`, with the body's UVs. Written as OBJ
    (Y up), so it goes through the same upload pipeline as any other file."""
    b = Body()
    a = b.anchors
    v, own = b.v, b.owner
    if kind == "tshirt":
        # cut by geometry (hem height, neckline, sleeve length), not by bone ownership: ownership
        # borders are ragged and leave holes where torso, arm and leg weights meet
        # (the neck bone also owns the trapezius/collarbone skin, so it is NOT excluded: the neckline cut does it)
        not_head_hands = ~np.isin(own, [15, 22, 23, 24, 18, 19, 20, 21] + list(range(25, 55)))
        hem_y, neck_r, cuff_x = a["hips_y"] - 0.02, 0.075, a["shoulder_x"] + 0.16
        nk = (v - a_neck(b))[:, [0, 2]]
        neck_zone = v[:, 1] > a["shoulder_y"] - 0.02          # the neckline is only cut above the shoulders
        inside_cuts = (v[:, 1] > hem_y) & ~(neck_zone & (np.linalg.norm(nk, axis=1) < neck_r)) \
            & (np.abs(v[:, 0]) < cuff_x) & (v[:, 1] < a["neck_y"] + 0.02)
        # faces crossing a cut are kept and their outside vertices clamped onto the cut -> clean edges
        face_mask = not_head_hands[b.f].all(1) & inside_cuts[b.f].any(1)
        v = v.copy()
        out = np.zeros(len(v), bool)
        out[np.unique(b.f[face_mask])] = True
        out &= ~inside_cuts
        v[out, 1] = np.maximum(v[out, 1], hem_y)
        v[out, 0] = np.clip(v[out, 0], -cuff_x, cuff_x)
        on = out & neck_zone
        r = np.linalg.norm(nk[on], axis=1)
        nk_on = nk[on] * (np.maximum(r, neck_r) / np.maximum(r, 1e-9))[:, None]
        v[on, 0], v[on, 2] = a_neck(b)[0] + nk_on[:, 0], a_neck(b)[2] + nk_on[:, 1]
    elif kind == "trousers":
        keep = np.isin(own, [0, 1, 2, 4, 5]) & (v[:, 1] < a["waist_y"] + 0.02) & (v[:, 1] > a["ankle_y"] + 0.04)
        face_mask = keep[b.f].all(1)
    else:
        raise ValueError(f"unknown test garment '{kind}'")
    tri = b.f[face_mask]
    # largest connected piece only
    parent = np.arange(len(v))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for t in tri:
        r = [find(i) for i in t]
        for i in r[1:]:
            parent[i] = r[0]
    roots = np.array([find(i) for i in tri[:, 0]])
    main = np.bincount(roots).argmax()
    sel = roots == main
    tri, tri_uv = tri[sel], b.ft[face_mask][sel]
    pos = v + b.normals(b.v) * ease   # offset along the unclamped body's normals (clamping skews them)
    used = np.unique(tri)
    remap = -np.ones(len(v), np.int64)
    remap[used] = np.arange(len(used))
    used_uv = np.unique(tri_uv)
    remap_uv = -np.ones(len(b.vt), np.int64)
    remap_uv[used_uv] = np.arange(len(used_uv))
    with open(out_path, "w", encoding="utf-8") as fo:
        fo.write(f"# VTryOn test garment: {kind} (SMPL-X template body, ease {ease} m)\n")
        for p in pos[used]:
            fo.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for t in b.vt[used_uv]:
            fo.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        for t, tu in zip(remap[tri] + 1, remap_uv[tri_uv] + 1):
            fo.write(f"f {t[0]}/{tu[0]} {t[1]}/{tu[1]} {t[2]}/{tu[2]}\n")
    return len(used), len(tri)


# ---------------- import + canonical mesh ----------------
def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_any(path: str):
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".blend":
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = list(src.objects)
        for o in dst.objects:
            if o is not None:
                bpy.context.scene.collection.objects.link(o)
    else:
        raise ValueError(f"Unsupported format '{ext}'")
    return [o for o in set(bpy.data.objects) - before if o.type == "MESH"]


def canonical_mesh(objs):
    """Evaluated, world-space triangles of all mesh objects -> (P, T, UV per corner, mat index, materials)."""
    dg = bpy.context.evaluated_depsgraph_get()
    Ps, Ts, UVs, Ms, mats, off = [], [], [], [], [], 0
    for o in objs:
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        n, nt = len(me.vertices), len(me.loop_triangles)
        if n == 0 or nt == 0:
            ev.to_mesh_clear()
            continue
        co = np.empty(n * 3)
        me.vertices.foreach_get("co", co)
        M = np.array(o.matrix_world)
        co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        tv = np.empty(nt * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tv)
        tl = np.empty(nt * 3, np.int64)
        me.loop_triangles.foreach_get("loops", tl)
        mi = np.empty(nt, np.int64)
        me.loop_triangles.foreach_get("material_index", mi)
        if me.uv_layers.active:
            uv = np.empty(len(me.loops) * 2)
            me.uv_layers.active.data.foreach_get("uv", uv)
            uv = uv.reshape(-1, 2)[tl].reshape(-1, 3, 2)
        else:
            uv = np.zeros((nt, 3, 2))
        slot_mats = [s.material for s in o.material_slots]
        base = len(mats)
        mats += slot_mats if slot_mats else [None]
        Ps.append(co)
        Ts.append(tv.reshape(-1, 3) + off)
        UVs.append(uv)
        Ms.append(np.clip(mi, 0, max(0, len(slot_mats) - 1)) + base)
        off += n
        ev.to_mesh_clear()
    if not Ps:
        raise ValueError("Missing geometry: the file contains no mesh faces")
    return np.concatenate(Ps), np.concatenate(Ts), np.concatenate(UVs), np.concatenate(Ms), mats


def weld(P, T, tol=1e-5):
    """Merge vertices at the same position (UV seams split them; the cloth needs one connected sheet)."""
    key = np.round(P / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    T2 = inv.ravel()[T]
    good = (T2[:, 0] != T2[:, 1]) & (T2[:, 1] != T2[:, 2]) & (T2[:, 0] != T2[:, 2])
    return P[idx], T2, good


def blender_to_smpl(P):
    return np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1)


def smpl_to_blender(P):
    return np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)


# ---------------- analysis / fitting ----------------
def fix_units(P, warnings):
    h = np.ptp(P[:, 1])
    if h <= 0:
        raise ValueError("Garment has no height (flat/degenerate geometry)")
    s = 1.0
    while h * s > 2.5:
        s /= 10.0
    while h * s < 0.2:
        s *= 10.0
    if s != 1.0:
        warnings.append(f"Rescaled by {s:g} (file units were not metres)")
    return P * s


def fix_orientation(P, warnings, rotate180=False):
    ext = np.ptp(P, axis=0)
    if ext[2] > 1.3 * ext[1] and ext[2] > ext[0] * 0.6:  # lying flat: depth is the largest extent
        P = np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1)
        warnings.append("Garment was lying down; stood upright (check front/back in the preview)")
    if rotate180:
        P = P * [-1, 1, -1]
    return P


def guess_category(P, name: str, hint: str = ""):
    if hint:
        return hint, "given"
    low = name.lower()
    for k, cat in KEYWORDS:
        if k in low:
            return cat, f"name contains '{k}'"
    h, w = np.ptp(P[:, 1]), np.ptp(P[:, 0])
    bottom = P[P[:, 1] < P[:, 1].min() + 0.08 * h]
    xs = np.sort(bottom[:, 0])
    legs = len(xs) > 10 and np.max(np.diff(xs)) > 0.05
    if legs:
        return ("jumpsuit" if h > 1.25 else "trousers"), "two leg openings"
    if w > 1.05 * h or h < 0.85:
        return "tshirt", "upper-body proportions"
    return "dress", "long one-piece"


def signed_distances(P, tree):
    out = np.empty(len(P))
    for i, p in enumerate(P):
        loc, nrm, _, _ = tree.find_nearest(p.tolist())
        out[i] = (p - np.array(loc)) @ np.array(nrm) if loc is not None else 1.0
    return out


def place(P, region, a, body: "Body", warnings):
    """Put the garment on the body. A garment authored on this body (e.g. CLO3D on the base avatar)
    already sits right: keep it. Otherwise align its top to the neck/waist and centre it on the
    body's torso SURFACE (joints sit towards the back of the torso, so they are a bad depth centre)."""
    sd = signed_distances(P, body.bvh())
    if 0.0 <= np.median(sd) <= 0.08 and (sd < 0).mean() < 0.05:
        return P, "as authored (already on the body)"
    top = P[:, 1].max()
    dy = ((a["waist_y"] + 0.03) if region == "lower" else (a["neck_y"] + 0.03)) - top
    Q = P + [0, dy, 0]
    band = (body.posed[:, 1] > Q[:, 1].min()) & (body.posed[:, 1] < Q[:, 1].max()) & \
        np.isin(body.owner, [0, 1, 2, 3, 6, 9, 13, 14])
    bz = body.posed[band, 2]
    dz = (bz.min() + bz.max()) / 2 - (Q[:, 2].min() + Q[:, 2].max()) / 2
    warnings.append("Garment was not positioned on the body; auto-placed (check the preview)")
    return Q + [-(Q[:, 0].min() + Q[:, 0].max()) / 2, 0, dz], "auto-placed"


def sleeve_angle(P, a):
    """Degrees the garment's sleeves hang below horizontal (0 = T-pose)."""
    s = P[(np.abs(P[:, 0]) > a["shoulder_x"] + 0.06) & (P[:, 1] > a["hips_y"])]
    if len(s) < 20:
        return 0.0
    # slope of the sleeve's centre line (mean height per outward step), not the height of its
    # vertices: a sleeve's underside hangs below the shoulder even in a T-pose
    dx = np.abs(s[:, 0]) - a["shoulder_x"]
    bins = np.linspace(dx.min(), dx.max(), 6)
    cx, cy = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (dx >= lo) & (dx < hi)
        if m.sum() >= 3:
            cx.append(dx[m].mean())
            cy.append(s[m, 1].mean())
    if len(cx) < 2:
        return 0.0
    slope = np.polyfit(cx, cy, 1)[0]
    ang = float(np.clip(np.degrees(np.arctan(-slope)), 0.0, 70.0))
    return ang if ang >= 10.0 else 0.0


def conform(P, T, body: Body, ease: float, tol: float = 0.003):
    """Make the garment clear the body by `ease`: only vertices inside the body or closer than
    ease - tol are moved. The required DISPLACEMENT (not the geometry) is smoothed into the 2-ring
    around them, so corrections blend in without shrinking the design; untouched areas stay exactly
    as authored."""
    tree = body.bvh()
    nbr = [[] for _ in range(len(P))]
    for t in T:
        for i in range(3):
            nbr[t[i]] += [t[(i + 1) % 3], t[(i + 2) % 3]]
    nbr = [np.unique(x) if x else np.array([], np.int64) for x in nbr]
    P = P.copy()
    moved = np.zeros(len(P), bool)

    def violations(idx):
        delta = np.zeros((len(P), 3))
        hit = np.zeros(len(P), bool)
        for i in idx:
            loc, nrm, _, _ = tree.find_nearest(P[i].tolist())
            if loc is None:
                continue
            loc, n = np.array(loc), np.array(nrm)
            depth = (P[i] - loc) @ n
            if depth < ease - tol:
                delta[i] = n * (ease - depth)
                hit[i] = True
        return delta, hit

    idx = np.arange(len(P))
    for _ in range(4):
        delta, hit = violations(idx)
        if not hit.any():
            break
        moved |= hit
        region = hit.copy()
        for _ in range(2):                          # 2-ring neighbourhood
            for i in np.nonzero(region)[0]:
                region[nbr[i]] = True
        for _ in range(3):                          # diffuse the displacement field
            ri = np.nonzero(region & ~hit)[0]
            delta[ri] = np.array([delta[nbr[i]].mean(0) if len(nbr[i]) else delta[i] for i in ri])
        P += delta
        idx = np.nonzero(region)[0]
    inside = (signed_distances(P, tree) < 0).mean()
    return P, moved, inside


def pin_weights(P, region, a):
    w = np.full(len(P), 0.12)                                         # soft shape memory everywhere
    top = P[:, 1].max()
    if region == "lower":
        w[P[:, 1] > top - 0.07] = 1.0                                 # waistband
    else:
        yoke = (P[:, 1] > a["shoulder_y"] - 0.06) & (np.abs(P[:, 0]) < a["shoulder_x"] + 0.03)
        w[yoke] = 1.0                                                 # collar + shoulders
    w[P[:, 1] < P[:, 1].min() + 0.04] = 0.05                         # hems swing freely
    return w


def check_textures(mats, warnings):
    for m in mats:
        if m is None or not m.node_tree:
            continue
        for n in m.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image is not None:
                img = n.image
                if not img.packed_file and not (img.filepath and os.path.isfile(bpy.path.abspath(img.filepath))) \
                        and not img.has_data:
                    warnings.append(f"Missing texture '{img.name}' ({img.filepath}) in material '{m.name}'")


# ---------------- output ----------------
def build_object(gid, P, T, UV, MI, mats, pins, props):
    me = bpy.data.meshes.new(f"VFR_Garment_{gid}")
    me.from_pydata(P.tolist(), [], T.tolist())
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", UV.reshape(-1).astype(np.float32))
    for m in mats:
        me.materials.append(m)
    if mats:
        me.polygons.foreach_set("material_index", MI.astype(np.int32))
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()
    ob = bpy.data.objects.new(f"VFR_Garment_{gid}", me)
    vg = ob.vertex_groups.new(name="vfr_pin")
    for i, wv in enumerate(pins):
        vg.add([i], float(wv), "REPLACE")
    for k, v in props.items():
        ob[k] = v
    return ob


def default_material(category):
    m = bpy.data.materials.new(f"VFR_Fabric_{category}")
    m.diffuse_color = {"jeans": (0.12, 0.2, 0.42, 1), "trousers": (0.25, 0.22, 0.2, 1)}.get(category, (0.75, 0.12, 0.12, 1))
    if m.node_tree is None:
        m.use_nodes = True
    bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf:
        bsdf.inputs["Base Color"].default_value = m.diffuse_color
        bsdf.inputs["Roughness"].default_value = 0.85
    return m


def render_preview(ob, body: Body, path):
    sc = bpy.context.scene
    sc.collection.objects.link(ob)
    bm = bpy.data.meshes.new("preview_body")
    bm.from_pydata(smpl_to_blender(body.posed).tolist(), [], body.f.tolist())
    bm.polygons.foreach_set("use_smooth", [True] * len(bm.polygons))
    bo = bpy.data.objects.new("preview_body", bm)
    mat = bpy.data.materials.new("skin")
    mat.diffuse_color = (0.55, 0.55, 0.58, 1)
    bm.materials.append(mat)
    sc.collection.objects.link(bo)
    lo, hi = np.array(ob.bound_box[0]), np.array(ob.bound_box[6])
    c = (lo + hi) / 2
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = float(max(hi[0] - lo[0], hi[2] - lo[2]) * 1.25)
    cam.location = (c[0], -4.0, c[2])
    cam.rotation_euler = (math.pi / 2, 0, 0)
    sc.collection.objects.link(cam)
    sc.camera = cam
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.rotation_euler = (math.radians(50), 0, math.radians(20))
    sc.collection.objects.link(sun)
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 12
    sc.render.resolution_x, sc.render.resolution_y = 320, 400
    sc.render.film_transparent = True
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    sc.collection.objects.unlink(bo)
    sc.collection.objects.unlink(cam)
    sc.collection.objects.unlink(sun)
    sc.collection.objects.unlink(ob)


def process(args) -> dict:
    reg = Registry(args.library)
    pend = reg.pending_dir(args.id)
    os.makedirs(pend, exist_ok=True)
    fabrics = load_fabrics()
    rep = {"id": args.id, "original": os.path.abspath(args.input), "status": "error", "errors": [],
           "warnings": [], "stats": {}, "uploaded": time.strftime("%Y-%m-%d %H:%M:%S")}
    name = args.name or os.path.splitext(os.path.basename(args.input))[0].replace("_", " ")
    rep["name"] = name
    t0 = time.perf_counter()
    try:
        ext = os.path.splitext(args.input)[1].lower()
        if ext in IMAGE_FORMATS:
            raise ValueError("This is a photograph. A photo has no 3D garment geometry and needs a separate "
                             "3D reconstruction step (not available yet); upload an FBX/OBJ/GLB/glTF/.blend garment.")
        clear_scene()
        try:
            objs = import_any(args.input)
        except RuntimeError as exc:
            raise ValueError(f"Could not read the file (corrupted or wrong format): {str(exc).splitlines()[0]}")
        if not objs:
            raise ValueError("Missing geometry: the file contains no mesh objects")
        P, T, UV, MI, mats = canonical_mesh(objs)
        check_textures([m for m in mats if m], rep["warnings"])
        n_raw = len(P)
        P, T, good = weld(P, T)
        if (~good).sum():
            rep["warnings"].append(f"Removed {(~good).sum()} degenerate triangles")
        T, UV, MI = T[good], UV[good], MI[good]
        if len(T) == 0:
            raise ValueError("Missing geometry: no valid faces after cleanup")
        # non-manifold edges (shared by > 2 triangles) make cloth unstable
        e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
        _, cnt = np.unique(e, axis=0, return_counts=True)
        if (cnt > 2).sum():
            rep["warnings"].append(f"{(cnt > 2).sum()} non-manifold edges (may simulate poorly)")
        P = blender_to_smpl(P)
        P = fix_units(P, rep["warnings"])
        P = fix_orientation(P, rep["warnings"], args.rotate180)
        if len(P) > MAX_VERTS:
            rep["warnings"].append(f"{len(P)} vertices exceeds the real-time cloth budget ({MAX_VERTS}); "
                                   "decimate the garment for live use")
        cat, why = guess_category(P, os.path.basename(args.input), args.category)
        region = next((r for r, cats in fabrics["regions"].items() if cat in cats), "upper")
        rep.update(category=cat, region=region, category_reason=why)
        body = Body()
        a = body.anchors
        P, placement = place(P, region, a, body, rep["warnings"])
        rep["placement"] = placement
        angle = sleeve_angle(P, a) if region in ("upper", "full") else 0.0
        body.pose_arms(angle)
        ease = fabrics["fit_styles"].get(args.fit, 0.012)
        P, moved, inside = conform(P, T, body, ease)
        rep["stats"].update(vertices=int(len(P)), raw_vertices=int(n_raw), triangles=int(len(T)),
                            height_m=round(float(np.ptp(P[:, 1])), 3), width_m=round(float(np.ptp(P[:, 0])), 3),
                            moved_to_fit_pct=round(100 * moved.mean(), 1), inside_body_pct=round(100 * inside, 2),
                            sleeve_angle_deg=round(angle, 1))
        if inside > 0.03:
            raise ValueError(f"Fitting failed: {100 * inside:.1f}% of the garment stays inside the body "
                             "(wrong category/placement?)")
        if moved.mean() > 0.6:
            rep["warnings"].append(f"{100 * moved.mean():.0f}% of the garment had to be moved to fit: it may be "
                                   "much smaller than the body or the wrong category")
        pins = pin_weights(P, region, a)
        if (pins >= 1.0).sum() < 10:
            raise ValueError("No attachment region found (garment does not reach the shoulders/waist)")
        keep = [m for m in mats if m]
        if keep:  # drop empty slots, remap face material indices onto the kept ones
            new_index = {i: keep.index(m) for i, m in enumerate(mats) if m}
            MI = np.array([new_index.get(int(i), 0) for i in MI])
            mats = keep
        else:
            mats, MI = [default_material(cat)], np.zeros(len(T), np.int64)
        for m in mats:  # make the processed asset self-contained
            for nd in (m.node_tree.nodes if m.node_tree else []):
                if nd.type == "TEX_IMAGE" and nd.image is not None and nd.image.has_data and not nd.image.packed_file:
                    try:
                        nd.image.pack()
                    except RuntimeError:
                        pass
        profile = fabrics["category_defaults"].get(cat, "cotton_jersey")
        rep.update(fabric_profile=profile, fit_style=args.fit, materials=[m.name for m in mats],
                   object=f"VFR_Garment_{args.id}", bind_pose={"shoulder_drop_deg": round(angle, 2)})
        props = {"vfr_id": args.id, "vfr_region": region, "vfr_category": cat, "vfr_ease": ease,
                 "vfr_shoulder_drop_deg": angle}
        ob = build_object(args.id, smpl_to_blender(P), T, UV, MI, mats, pins, props)
        try:
            render_preview(ob, body, os.path.join(pend, "preview.png"))
        except Exception as exc:  # preview is optional
            rep["warnings"].append(f"Preview render failed: {exc}")
        bpy.data.libraries.write(os.path.join(pend, "garment.blend"), {ob}, fake_user=True, compress=True)
        rep["status"] = "ok"
    except ValueError as exc:
        rep["errors"].append(str(exc))
    except Exception:
        rep["errors"].append("Internal processing error:\n" + traceback.format_exc())
    rep["stats"]["processing_s"] = round(time.perf_counter() - t0, 2)
    with open(os.path.join(pend, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    return rep


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--id")
    ap.add_argument("--name", default="")
    ap.add_argument("--category", default="")
    ap.add_argument("--fit", default="regular")
    ap.add_argument("--library", default=os.path.join(ROOT, "assets", "garments"))
    ap.add_argument("--rotate180", action="store_true")
    ap.add_argument("--generate")
    ap.add_argument("--out")
    ap.add_argument("--ease", type=float, default=0.012)
    a = ap.parse_args(argv)
    if a.generate:
        nv, nt = generate(a.generate, a.out, a.ease)
        print(f"VFR_GENERATED {a.generate}: {nv} vertices, {nt} triangles -> {a.out}")
        return
    rep = process(a)
    print("VFR_REPORT " + json.dumps({k: rep[k] for k in ("id", "status", "errors", "warnings", "stats")}))


if __name__ == "__main__":
    main()
