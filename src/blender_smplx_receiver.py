"""
SMPL-X digital twin for Blender, driven live by the VFR tracking service, composited over the
live camera feed.

What it builds (once; re-running the script reuses it):
  VFR_World        empty, +90 deg about X: converts SMPL-X Y-up space into Blender's Z-up world.
  VFR_Camera       child of VFR_World at the origin; intrinsics from the tracking packet
                   (config/camera_intrinsics.json via scripts/calibrate_camera.py, else 60 deg FOV).
  SMPLX_Twin_rig   55-bone armature from models/smplx/SMPLX_NEUTRAL.npz, placed where the person
                   stands relative to the camera ("transl" in the packet). Bones have identity rest
                   orientation, so SMPL-X parent-local rotations apply 1:1. Native Limit Rotation
                   constraints (config/body_model.json) act as a safety net behind the solver's limits.
  SMPLX_Twin       skinned mesh + "VFR_PoseCorrectives" shape key (SMPL-X pose blend shapes,
                   recomputed per pose so elbows/shoulders keep their volume like the real model).
  VFR_CameraFeed   one image datablock, refreshed in place from shared memory (src/camera/frame_share.py).
                   Shown unlit as the camera's background in the viewport, and on VFR_Background
                   (emission plane behind the person) for final renders.

Run: blender --python src/blender_smplx_receiver.py   (or the dashboard's "Open Digital Twin" button)
Env: VFR_ROOT (project folder), VFR_BIND (127.0.0.1), VFR_PORT (8766).
Pose arrives over TCP (newline-delimited JSON, key "smplx"); frames via shared memory.
"""
import os
import sys
import json
import math
import socket
import time

import bpy
import numpy as np
from mathutils import Quaternion

HOST = os.environ.get("VFR_BIND", "127.0.0.1")
PORT = int(os.environ.get("VFR_PORT", "8766"))
KEY = "VFR_SMPLX_RECEIVER"
NAME = "SMPLX_Twin"
VERSION = 2            # bump when the scene layout changes; older twins are rebuilt
BG_DISTANCE = 12.0     # m behind the camera's near plane; must be farther than the person
DEFAULT_TRANSL = (0.0, 0.0, 2.5)   # camera frame (x right, y down, z forward) until tracking says otherwise

JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "jaw", "left_eye_smplhf", "right_eye_smplhf",
    "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]


# ---------------- paths / logging ----------------
def project_root() -> str:
    cands = [os.environ.get("VFR_ROOT")]
    f = globals().get("__file__")
    if f and os.path.isfile(f):
        cands.append(os.path.dirname(os.path.dirname(os.path.abspath(f))))
    for t in bpy.data.texts:  # opened in Blender's text editor
        if t.filepath.endswith("blender_smplx_receiver.py"):
            cands.append(os.path.dirname(os.path.dirname(bpy.path.abspath(t.filepath))))
    if bpy.data.filepath:
        cands.append(os.path.dirname(bpy.data.filepath))
    for root in filter(None, cands):
        if os.path.isfile(os.path.join(root, "models", "smplx", "SMPLX_NEUTRAL.npz")):
            return root
    raise FileNotFoundError("SMPLX_NEUTRAL.npz not found; set VFR_ROOT to the project folder")


ROOT = project_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.camera.frame_share import FrameReader  # noqa: E402  (numpy + stdlib only)
from src.camera.body_model import load_config, joint_limits  # noqa: E402


def _log(msg: str):
    """Blender's console is hidden on Windows; mirror to outputs/logs/blender_twin.log."""
    print(msg)
    try:
        path = os.path.join(ROOT, "outputs", "logs", "blender_twin.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    except Exception:
        pass


# ---------------- scene construction ----------------
def _remove_default_cube():
    """Remove Blender's untouched startup cube and camera (VFR_Camera replaces it). The startup
    light stays: it lights the twin in rendered views."""
    cube = bpy.data.objects.get("Cube")
    if cube is not None and cube.type == "MESH" and cube.data.name == "Cube" and len(cube.data.vertices) == 8:
        bpy.data.objects.remove(cube, do_unlink=True)
    cam = bpy.data.objects.get("Camera")
    if cam is not None and cam.type == "CAMERA" and cam.data.name == "Camera":
        bpy.data.objects.remove(cam, do_unlink=True)


def _remove_old_twin():
    for n in (NAME, NAME + "_rig", "VFR_Background", "VFR_Camera", "VFR_World"):
        o = bpy.data.objects.get(n)
        if o is not None:
            bpy.data.objects.remove(o, do_unlink=True)


def _set_object_mode():
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def build_scene():
    """Create (or reuse) world root, camera, background and the rigged twin. Returns the rig."""
    arm = bpy.data.objects.get(NAME + "_rig")
    if arm is not None and arm.get("vfr_version") == VERSION and bpy.data.objects.get(NAME):
        return arm
    _set_object_mode()
    _remove_old_twin()
    _remove_default_cube()
    scene = bpy.context.scene

    world = bpy.data.objects.new("VFR_World", None)
    world.rotation_euler = (math.pi / 2, 0.0, 0.0)  # SMPL-X / camera-space (Y up) -> Blender (Z up)
    scene.collection.objects.link(world)

    cam = bpy.data.objects.new("VFR_Camera", bpy.data.cameras.new("VFR_Camera"))
    cam.parent = world  # at the origin, looking down SMPL -Z (towards the person) with +Y up
    cam.data.clip_start, cam.data.clip_end = 0.05, BG_DISTANCE + 5
    scene.collection.objects.link(cam)
    scene.camera = cam

    img = bpy.data.images.get("VFR_CameraFeed") or bpy.data.images.new("VFR_CameraFeed", 640, 480, alpha=False)
    img.colorspace_settings.name = "sRGB"
    bg = cam.data.background_images.new()
    bg.image, bg.display_depth, bg.frame_method, bg.alpha = img, "BACK", "FIT", 1.0
    cam.data.show_background_images = True

    plane = _build_background_plane(img)
    plane.parent = cam
    scene.collection.objects.link(plane)
    plane.hide_viewport = True  # viewport uses the (unlit, exact) camera background; renders use the plane

    arm = _build_twin(scene)
    arm.parent = world
    arm["vfr_version"] = VERSION
    apply_camera(cam, plane, None)
    set_transl(arm, DEFAULT_TRANSL)
    return arm


def _build_background_plane(img):
    me = bpy.data.meshes.new("VFR_Background")
    me.from_pydata([(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name="UVMap")
    for li, (u, v) in zip(range(4), [(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[li].uv = (u, v)
    mat = bpy.data.materials.get("VFR_CameraFeed") or bpy.data.materials.new("VFR_CameraFeed")
    if mat.node_tree is None:  # Blender < 5 creates node-less materials
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image, tex.interpolation = img, "Closest"  # 1 texel = 1 camera pixel; no extra filtering
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    me.materials.append(mat)
    plane = bpy.data.objects.new("VFR_Background", me)
    plane.visible_shadow = False
    return plane


def _build_twin(scene):
    d = np.load(os.path.join(ROOT, "models", "smplx", "SMPLX_NEUTRAL.npz"), allow_pickle=True)
    v = d["v_template"].astype(np.float64)
    faces = d["f"].astype(np.int64)
    W = d["weights"]
    parents = d["kintree_table"][0].astype(np.int64)
    J = d["J_regressor"].astype(np.float64) @ v

    mesh = bpy.data.meshes.new(NAME)
    mesh.from_pydata(v.tolist(), [], faces.tolist())
    mesh.update()
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    body = bpy.data.objects.new(NAME, mesh)

    arm_data = bpy.data.armatures.new(NAME + "_rig")
    arm_data.display_type = "STICK"
    arm = bpy.data.objects.new(NAME + "_rig", arm_data)
    scene.collection.objects.link(arm)
    scene.collection.objects.link(body)

    bpy.context.view_layer.update()  # drop dangling entries of objects removed above
    for o in bpy.context.view_layer.objects:
        if o is not None:
            o.select_set(False)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        ebs = []
        for i, name in enumerate(JOINT_NAMES):
            eb = arm_data.edit_bones.new(name)
            eb.head = J[i].tolist()
            eb.tail = (J[i] + [0.0, 0.03, 0.0]).tolist()  # +Y, roll 0 -> identity rest orientation
            eb.roll = 0.0
            if i > 0:
                eb.parent = ebs[parents[i]]
                eb.use_connect = False
            ebs.append(eb)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")  # never leave the rig frozen in edit mode

    for i, name in enumerate(JOINT_NAMES):
        vg = body.vertex_groups.new(name=name)
        for vi in np.nonzero(W[:, i] > 1e-4)[0]:
            vg.add([int(vi)], float(W[vi, i]), "REPLACE")
    body.parent = arm
    body.modifiers.new("Armature", "ARMATURE").object = arm

    # SMPL-X pose correctives as a shape key (evaluated before the armature, like SMPL-X itself)
    body.shape_key_add(name="Basis", from_mix=False)
    sk = body.shape_key_add(name="VFR_PoseCorrectives", from_mix=False)
    sk.value = 1.0

    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    _add_limit_constraints(arm)
    arm.select_set(False)
    _log(f"[VFR] Built {NAME}: {len(v)} verts, {len(JOINT_NAMES)} bones, pose correctives + joint limits")
    return arm


def _add_limit_constraints(arm):
    """Native Limit Rotation per body joint, same ranges/Euler order as the solver (safety net)."""
    lims = joint_limits(load_config())
    for name, lim in lims.items():
        pb = arm.pose.bones.get(name)
        if pb is None:
            continue
        c = pb.constraints.new("LIMIT_ROTATION")
        c.name = "VFR_Anatomy"
        c.owner_space = "LOCAL"
        c.euler_order = lim["order"]
        c.use_limit_x = c.use_limit_y = c.use_limit_z = True
        c.min_x, c.min_y, c.min_z = lim["lo"].tolist()
        c.max_x, c.max_y, c.max_z = lim["hi"].tolist()


# ---------------- per-packet updates ----------------
class Correctives:
    """SMPL-X pose blend shapes: offsets = (R_j - I).flatten() @ posedirs, for joints 1..54.

    The 486x31k product costs ~45 ms in Blender's numpy, so: body joints (dense) + hand joints
    (sparse: they only touch hand vertices) are computed on a worker thread (numpy releases the
    GIL); jaw/eyes are always zero and skipped. The main thread only writes finished results
    (~1 ms), one tick behind the pose at most."""
    BODY_F = 21 * 9          # feature rows for body joints 1..21
    HAND_J0 = 24             # rvecs index of left_index1 (after jaw + 2 eyes)

    def __init__(self, body):
        import threading
        d = np.load(os.path.join(ROOT, "models", "smplx", "SMPLX_NEUTRAL.npz"), allow_pickle=True)
        pd = d["posedirs"].astype(np.float32)                     # (V, 3, 486)
        P = pd.reshape(-1, pd.shape[-1]).T                        # (486, V*3), as in smplx.lbs
        self.P_body = np.ascontiguousarray(P[:self.BODY_F])
        self.hands = []
        for j in range(self.HAND_J0, 54):
            rows = P[j * 9:(j + 1) * 9]
            cols = np.nonzero(np.abs(rows).max(axis=0) > 1e-5)[0]
            if len(cols):
                self.hands.append((j, cols, np.ascontiguousarray(rows[:, cols])))
        self.base = d["v_template"].astype(np.float32).ravel()
        self.key = body.data.shape_keys.key_blocks["VFR_PoseCorrectives"]
        self._lock = threading.Lock()
        self._job = None      # latest rvecs waiting to be computed
        self._result = None   # finished offsets waiting to be written
        self._last = None
        self._wake = threading.Event()
        self._alive = True
        threading.Thread(target=self._worker, name="VFR-Correctives", daemon=True).start()

    def offsets(self, rvecs_54: np.ndarray) -> np.ndarray:
        feat = (self.rotmats(rvecs_54) - np.eye(3)[None]).reshape(-1).astype(np.float32)
        off = feat[:self.BODY_F] @ self.P_body
        for j, cols, Pj in self.hands:
            off[cols] += feat[j * 9:(j + 1) * 9] @ Pj
        return off

    def _worker(self):
        while self._alive:
            self._wake.wait(0.5)
            self._wake.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is not None:
                off = self.offsets(job)
                with self._lock:
                    self._result = off

    def submit(self, rvecs_54: np.ndarray):
        if self._last is not None and np.abs(rvecs_54 - self._last).max() < 1e-4:
            return  # pose unchanged: keep the current shape key
        self._last = rvecs_54.copy()
        with self._lock:
            self._job = rvecs_54.copy()
        self._wake.set()

    def flush(self) -> bool:
        """Main thread: write a finished result into the shape key. Returns True if written."""
        with self._lock:
            off, self._result = self._result, None
        if off is None:
            return False
        self.key.data.foreach_set("co", self.base + off)
        self.key.id_data.update_tag()
        return True

    def stop(self):
        self._alive = False
        self._wake.set()

    @staticmethod
    def rotmats(rvecs: np.ndarray) -> np.ndarray:
        th = np.linalg.norm(rvecs, axis=1, keepdims=True)
        k = rvecs / np.maximum(th, 1e-12)
        K = np.zeros((len(rvecs), 3, 3))
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
        K = K - K.transpose(0, 2, 1)
        s, c = np.sin(th)[:, :, None], np.cos(th)[:, :, None]
        return np.eye(3)[None] + s * K + (1 - c) * (K @ K)

    def apply_now(self, rvecs_54: np.ndarray):
        """Synchronous version (tests / one-off)."""
        self.key.data.foreach_set("co", self.base + self.offsets(rvecs_54))
        self.key.id_data.update_tag()


def _quat(rv) -> Quaternion:
    rv = np.asarray(rv, dtype=np.float64)
    ang = float(np.linalg.norm(rv))
    if ang < 1e-8:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    return Quaternion((rv / ang).tolist(), ang)


def apply_pose(arm, p: dict, correctives=None) -> bool:
    """packet['smplx'] -> pose bones (+ correctives, placement). Ignores malformed packets."""
    go, bp = p.get("global_orient", []), p.get("body_pose", [])
    lh, rh = p.get("left_hand_pose", [0.0] * 45), p.get("right_hand_pose", [0.0] * 45)
    if len(go) != 3 or len(bp) != 63 or len(lh) != 45 or len(rh) != 45:
        return False
    rots = np.array([go] + [bp[i:i + 3] for i in range(0, 63, 3)] + [[0.0] * 3] * 3
                    + [lh[i:i + 3] for i in range(0, 45, 3)] + [rh[i:i + 3] for i in range(0, 45, 3)],
                    dtype=np.float64)
    if not np.all(np.isfinite(rots)):
        return False
    pbs = arm.pose.bones
    for name, rv in zip(JOINT_NAMES, rots):
        pbs[name].rotation_quaternion = _quat(rv)
    if correctives is not None:
        correctives.submit(rots[1:])
    t = p.get("transl")
    if t is not None and len(t) == 3 and all(math.isfinite(x) for x in t):
        set_transl(arm, t)
    return True


def set_transl(arm, t_cam):
    """Camera frame (x right, y down, z forward) -> VFR_World / SMPL frame (x, -y, -z)."""
    arm.location = (float(t_cam[0]), -float(t_cam[1]), -float(t_cam[2]))


def apply_camera(cam, plane, K):
    """Match the Blender camera to the physical (mirrored) camera intrinsics and size the
    background plane to fill the frustum exactly at BG_DISTANCE."""
    if K is None:
        w, h = 640, 480
        f = (w / 2) / math.tan(math.radians(60) / 2)
        K = {"w": w, "h": h, "fx": f, "fy": f, "cx": w / 2, "cy": h / 2}
    w, h = int(K["w"]), int(K["h"])
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = 1.0, float(K["fx"]) / float(K["fy"])
    cd = cam.data
    cd.sensor_fit, cd.sensor_width = "HORIZONTAL", 36.0
    cd.lens = float(K["fx"]) * 36.0 / w
    cd.shift_x = (w / 2 - float(K["cx"])) / w        # principal point offset (units of image width)
    cd.shift_y = (float(K["cy"]) - h / 2) / w
    full_w = BG_DISTANCE * w / float(K["fx"])
    full_h = BG_DISTANCE * h / float(K["fy"])
    plane.location = (cd.shift_x * full_w, cd.shift_y * full_w, -BG_DISTANCE)
    plane.scale = (full_w, full_h, 1.0)
    cam["vfr_K"] = json.dumps(K, sort_keys=True)


class CameraFeed:
    """Copies the newest shared-memory frame into the single VFR_CameraFeed image."""
    STALE_S = 2.0

    def __init__(self):
        self.reader = FrameReader()
        self.img = bpy.data.images["VFR_CameraFeed"]
        self.rgba = None
        self.frames = 0
        self.last_new = 0.0
        self.status = "waiting"

    def update(self) -> bool:
        frame, _ = self.reader.read()
        now = time.time()
        if frame is None:
            if now - self.last_new > self.STALE_S:
                self.status = "no signal"
                self.reader.reset()  # writer may have restarted: re-attach next tick
            return False
        h, w = frame.shape[:2]
        if tuple(self.img.size) != (w, h):
            self.img.scale(w, h)
        if self.rgba is None or self.rgba.shape[:2] != (h, w):
            self.rgba = np.ones((h, w, 4), np.float32)
        # BGR top-row-first -> RGBA bottom-row-first; byte image, so values are taken as sRGB as-is
        np.multiply(frame[::-1, :, ::-1], 1.0 / 255.0, out=self.rgba[:, :, :3], casting="unsafe")
        self.img.pixels.foreach_set(self.rgba.ravel())
        self.img.update()
        self.frames += 1
        self.last_new = now
        self.status = "live"
        return True


def show_camera_view():
    """Look through VFR_Camera in every 3D viewport, framed to fill it (no-op in background mode)."""
    cam = bpy.data.objects.get("VFR_Camera")
    if cam is not None:
        cam.data.passepartout_alpha = 1.0  # hide everything outside the camera frame
    wm = bpy.context.window_manager
    for win in (wm.windows if wm else []):
        for area in win.screen.areas:
            if area.type != "VIEW_3D":
                continue
            sp = area.spaces.active
            sp.region_3d.view_perspective = "CAMERA"
            sp.shading.type = "SOLID"
            sp.overlay.show_floor = sp.overlay.show_axis_x = sp.overlay.show_axis_y = False
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region is not None:
                with bpy.context.temp_override(window=win, area=area, region=region):
                    bpy.ops.view3d.view_center_camera()


def _redraw():
    wm = bpy.context.window_manager
    for win in (wm.windows if wm else []):
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# ---------------- network + main loop ----------------
def _stop_previous():
    for k in (KEY, "VFR_POSE_RECEIVER"):  # also the legacy landmark receiver on the same port
        old = bpy.app.driver_namespace.get(k)
        if old:
            old["active"] = False
            for sk in ("client", "server"):
                try:
                    old[sk].close()
                except Exception:
                    pass
            try:
                old["feed"].reader.reset()
                old["correctives"].stop()
            except Exception:
                pass


def start():
    _stop_previous()
    arm = build_scene()
    cam, plane = bpy.data.objects["VFR_Camera"], bpy.data.objects["VFR_Background"]
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    server.setblocking(False)
    state = {"active": True, "server": server, "client": None, "buffer": b"",
             "packets": 0, "applied": 0, "last_error": None, "rig": arm.name,
             "correctives": Correctives(bpy.data.objects[NAME]), "feed": CameraFeed(),
             "tick_ms": 0.0, "penetrations": {}}
    bpy.app.driver_namespace[KEY] = state

    def tick():
        s = bpy.app.driver_namespace.get(KEY)
        if s is None or not s["active"]:
            return None
        t0 = time.perf_counter()
        changed = False
        try:
            changed |= s["feed"].update()
        except Exception as exc:  # camera feed problems must never stop pose updates
            s["last_error"] = f"feed: {exc}"
        if s["client"] is None:
            try:
                c, addr = s["server"].accept()
                c.setblocking(False)
                s["client"] = c
                _log(f"[VFR] Tracking service connected: {addr}")
            except (BlockingIOError, OSError):
                pass
        if s["client"] is not None:
            try:
                while True:  # drain everything waiting; only the newest pose matters
                    data = s["client"].recv(1 << 20)
                    if not data:
                        raise ConnectionResetError("closed")
                    s["buffer"] += data
            except BlockingIOError:
                pass
            except OSError as exc:
                _log(f"[VFR] Tracking service disconnected: {exc}")
                try:
                    s["client"].close()
                except Exception:
                    pass
                s["client"], s["buffer"] = None, b""
        if len(s["buffer"]) > 16 * 1024 * 1024:
            s["buffer"], s["last_error"] = b"", "oversized buffer discarded"

        latest = None
        while b"\n" in s["buffer"]:
            line, s["buffer"] = s["buffer"].split(b"\n", 1)
            try:
                pkt = json.loads(line)
                s["packets"] += 1
                if isinstance(pkt, dict) and isinstance(pkt.get("smplx"), dict):
                    latest = pkt["smplx"]
            except (ValueError, UnicodeDecodeError) as exc:
                s["last_error"] = str(exc)
        if latest is not None:
            rig = bpy.data.objects.get(s["rig"])
            if rig is not None and apply_pose(rig, latest, s["correctives"]):
                s["applied"] += 1
                s["penetrations"] = latest.get("penetrations", {})
                K = latest.get("camera")
                if isinstance(K, dict) and cam.get("vfr_K") != json.dumps(K, sort_keys=True):
                    apply_camera(cam, plane, K)
                changed = True
        if s["correctives"].flush():
            changed = True
        if changed:
            _redraw()
        s["tick_ms"] = 0.9 * s["tick_ms"] + 0.1 * (time.perf_counter() - t0) * 1000
        return 1.0 / 60.0

    state["tick"] = tick  # timers don't run in `blender -b`; tests call this directly
    bpy.app.timers.register(tick, first_interval=0.05, persistent=True)
    _log(f"[VFR] SMPL-X twin receiver listening on {HOST}:{PORT}; camera feed via shared memory")


def start_safe():
    import traceback
    try:
        start()
        show_camera_view()
    except Exception:
        _log("[VFR] Twin FAILED:\n" + traceback.format_exc())
    return None  # one-shot timer


if __name__ == "__main__":
    if bpy.app.background:
        start_safe()
    else:
        # `blender --python` runs before the window is fully up; build once the UI is ready.
        bpy.app.timers.register(start_safe, first_interval=0.5)
