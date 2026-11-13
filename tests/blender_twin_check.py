"""End-to-end check of the Blender twin, run headless inside Blender.
    1. python tests/blender_twin_check.py               (venv: writes reference poses + vertices)
    2. blender -b --python tests/blender_twin_check.py  (builds scene, streams over TCP + shared memory)
Checks: full SMPL-X match incl. pose correctives, native joint-limit constraints == solver clamp,
live camera frames reach the image, intrinsics/placement applied, soak timing + memory."""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "outputs", "blender_twin_ref.json")
sys.path.insert(0, ROOT)

try:
    import bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

if not IN_BLENDER:
    import numpy as np
    import torch
    import smplx
    from src.camera.body_model import load_config, joint_limits, clamp_body_pose
    lims = joint_limits(load_config())
    rng = np.random.default_rng(1)
    layer = smplx.create(os.path.join(ROOT, "models"), model_type="smplx", gender="neutral",
                         ext="npz", use_pca=False, flat_hand_mean=True, num_betas=10)  # correctives ON
    bp = clamp_body_pose(rng.uniform(-0.6, 0.6, 63), lims)[0]
    pose = {"global_orient": rng.uniform(-0.6, 0.6, 3).tolist(), "body_pose": bp.tolist(),
            "left_hand_pose": rng.uniform(-0.4, 0.4, 45).tolist(),
            "right_hand_pose": rng.uniform(-0.4, 0.4, 45).tolist(), "betas": [0.0] * 10}
    with torch.no_grad():
        out = layer(**{k: torch.tensor(v, dtype=torch.float32)[None] for k, v in pose.items()})
    bad = np.zeros(63, np.float32)
    bad[3 * 3] = -0.8            # left knee bent forward (impossible)
    bad[17 * 3 + 1] = -0.9       # right elbow hyperextended
    bad[15 * 3 + 2] = 2.6        # left shoulder raised past its limit
    json.dump({"pose": pose, "verts": out.vertices[0].numpy().tolist(),
               "bad_pose": bad.tolist(), "bad_clamped": clamp_body_pose(bad, lims)[0].tolist()},
              open(REF, "w"))
    print("reference written:", REF)
    sys.exit(0)

# ---------------- inside Blender ----------------
import time  # noqa: E402
import ctypes  # noqa: E402
import socket  # noqa: E402
import numpy as np  # noqa: E402

os.environ.setdefault("VFR_ROOT", ROOT)
os.environ["VFR_PORT"] = "8799"  # don't collide with a live session
ns = {"__name__": "vfr_receiver", "__file__": os.path.join(ROOT, "src", "blender_smplx_receiver.py")}
exec(compile(open(ns["__file__"], encoding="utf-8").read(), ns["__file__"], "exec"), ns)
from src.camera.frame_share import FrameWriter  # noqa: E402
from src.camera.body_model import rotvec_to_mat  # noqa: E402


def rss_mb():
    if os.name != "nt":
        return float(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                   [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize",
                                                   "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                                   "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                                   "PagefileUsage", "PeakPagefileUsage")]
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    c = PMC()
    c.cb = ctypes.sizeof(c)
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 2**20


ref = json.load(open(REF))
writer = FrameWriter("vfr_camera_frame")  # stands in for the tracking service
ns["start"]()
state = bpy.app.driver_namespace[ns["KEY"]]
client = socket.create_connection(("127.0.0.1", 8799))
K = {"w": 640, "h": 480, "fx": 600.0, "fy": 600.0, "cx": 330.0, "cy": 236.0, "calibrated": True}


def send(smplx_params):
    """Send one pose and tick until it (and its background pose correctives) are applied."""
    client.sendall((json.dumps({"schema": 2, "points": [], "smplx": smplx_params}) + "\n").encode())
    for _ in range(100):
        state["tick"]()
        time.sleep(0.01)


body = bpy.data.objects[ns["NAME"]]
arm = bpy.data.objects[ns["NAME"] + "_rig"]
cam = bpy.data.objects["VFR_Camera"]
results = []

# 1. full SMPL-X match incl. pose correctives, placement + intrinsics
p = dict(ref["pose"], transl=[0.1, -0.2, 2.3], camera=K)
client.sendall(b"garbage-not-json\n")  # receiver must survive malformed lines
send(p)
assert state["applied"] >= 1, state
bpy.context.view_layer.update()
ev = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
inv = arm.matrix_world.inverted()
got = np.array([(inv @ (ev.matrix_world @ v.co))[:] for v in ev.data.vertices])
err = np.linalg.norm(got - np.array(ref["verts"]), axis=1)
print(f"[mesh] {len(got)} verts vs full SMPL-X (with pose correctives): mean {err.mean() * 1000:.2f} mm, max {err.max() * 1000:.2f} mm")
assert err.max() < 0.002
results.append(f"mesh == SMPL-X incl. correctives (max {err.max() * 1000:.2f} mm)")

loc = tuple(round(x, 4) for x in arm.location)
lens_ok = abs(cam.data.lens - 600 * 36 / 640) < 1e-4 and abs(cam.data.shift_x - (320 - 330) / 640) < 1e-6
print(f"[camera] lens {cam.data.lens:.3f} mm shift ({cam.data.shift_x:.4f},{cam.data.shift_y:.4f}); twin at {loc}")
assert loc == (0.1, 0.2, -2.3) and lens_ok
# the twin's pelvis must project where the solver put it: compare with pinhole projection
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
pel = arm.matrix_world @ arm.pose.bones["pelvis"].head
uv = world_to_camera_view(bpy.context.scene, cam, pel)
J0 = np.array(arm.data.bones["pelvis"].head_local)               # pelvis in SMPL frame (rest)
R0 = rotvec_to_mat(np.array(ref["pose"]["global_orient"]))
P = R0 @ J0 * [1, -1, -1] + np.array([0.1, -0.2, 2.3])            # camera frame (pelvis rotates about itself)
P = J0 * [1, -1, -1] + np.array([0.1, -0.2, 2.3])                  # root joint is the pivot -> unaffected by R0
u_exp, v_exp = (K["fx"] * P[0] / P[2] + K["cx"]) / 640, (K["fy"] * P[1] / P[2] + K["cy"]) / 480
px_err = np.hypot((uv.x - u_exp) * 640, ((1 - uv.y) - v_exp) * 480)
print(f"[camera] pelvis projects to ({uv.x * 640:.1f},{(1 - uv.y) * 480:.1f}) px, pinhole model ({u_exp * 640:.1f},{v_exp * 480:.1f}): {px_err:.3f} px")
assert px_err < 0.5
results.append(f"Blender camera == pinhole intrinsics ({px_err:.3f} px)")

# 2. native Limit Rotation constraints clamp exactly like the solver
bad = dict(ref["pose"], body_pose=ref["bad_pose"], left_hand_pose=[0.0] * 45, right_hand_pose=[0.0] * 45)
send(bad)
bpy.context.view_layer.update()
worst = 0.0
for i, name in enumerate(ns["JOINT_NAMES"][1:22]):
    pb = arm.pose.bones[name]
    par = pb.parent
    local = (par.matrix.to_3x3().inverted() @ pb.matrix.to_3x3()) if par else pb.matrix.to_3x3()
    exp = rotvec_to_mat(np.array(ref["bad_clamped"][i * 3:i * 3 + 3]))
    worst = max(worst, float(np.abs(np.array(local) - exp).max()))
print(f"[limits] Blender constraints vs solver clamp on an impossible pose: max matrix diff {worst:.2e}")
assert worst < 1e-4
results.append("native Limit Rotation == solver clamp")

# 3. live camera frames: content, orientation, colour order
img = bpy.data.images["VFR_CameraFeed"]
frame = np.zeros((480, 640, 3), np.uint8)
frame[:10, :20] = (255, 0, 0)  # top-left BLUE in BGR
writer.write(frame)
for _ in range(3):
    state["tick"]()
px = np.empty(640 * 480 * 4, np.float32)
img.pixels.foreach_get(px)
px = px.reshape(480, 640, 4)
top_left = px[-1, 0, :3]  # Blender rows are bottom-up
print(f"[feed] top-left pixel in Blender RGB {top_left.round(2)} (expect blue), status {state['feed'].status}")
assert np.allclose(top_left, [0, 0, 1], atol=1e-3) and state["feed"].status == "live"
results.append("camera frame reaches Blender image (orientation + RGB order)")

# 4. soak: 600 ticks with a new frame + pose each tick; timing + memory growth
rng = np.random.default_rng(0)
m0 = rss_mb()
ticks = []
for i in range(600):
    writer.write(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
    pp = dict(ref["pose"], body_pose=(np.array(ref["pose"]["body_pose"]) * np.cos(i / 20)).tolist())
    client.sendall((json.dumps({"smplx": pp}) + "\n").encode())
    time.sleep(0.001)
    t0 = time.perf_counter()
    state["tick"]()
    ticks.append((time.perf_counter() - t0) * 1000)
m1 = rss_mb()
print(f"[soak] 600 ticks (new 640x480 frame + pose each): tick p50 {np.median(ticks):.1f} ms, p95 {np.percentile(ticks, 95):.1f} ms;"
      f" frames {state['feed'].frames}, poses {state['applied']}; memory {m0:.0f} -> {m1:.0f} MB")
assert m1 - m0 < 50
results.append(f"soak: tick p50 {np.median(ticks):.1f} ms, memory +{m1 - m0:.0f} MB over 600 ticks")

# 5. camera disconnect: writer gone -> 'no signal', no crash, pose updates keep working
writer.close()
state["feed"].last_new -= 10
for _ in range(3):
    state["tick"]()
send(ref["pose"])
print(f"[feed] after camera disconnect: status '{state['feed'].status}', poses still applied {state['applied']}")
assert state["feed"].status == "no signal"
results.append("camera disconnect handled")

client.close()
print("\n".join(["BLENDER TWIN OK:"] + ["  - " + r for r in results]))
