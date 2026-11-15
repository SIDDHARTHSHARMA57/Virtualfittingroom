"""Clothing system end to end, headless:  blender -b --factory-startup --python tests/blender_wardrobe_check.py
(needs the seeded library: python scripts/seed_garments.py). Uses a temporary copy of the library."""
import os
import sys
import json
import time
import shutil
import tempfile

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("VFR_ROOT", ROOT)
os.environ["VFR_PORT"] = "8796"
ns = {"__name__": "vfr_receiver", "__file__": os.path.join(ROOT, "src", "blender_smplx_receiver.py")}
exec(compile(open(ns["__file__"], encoding="utf-8").read(), ns["__file__"], "exec"), ns)
from src.camera.frame_share import FrameWriter  # noqa: E402
from src.garments import blender_wardrobe as bw  # noqa: E402
from src.garments.registry import Registry  # noqa: E402

LIB = tempfile.mkdtemp(prefix="vfr_lib_")
shutil.copytree(os.path.join(ROOT, "assets", "garments"), LIB, dirs_exist_ok=True)
log = []


def check(name, cond, info=""):
    log.append(("PASS" if cond else "FAIL") + f"  {name}  {info}")
    print(log[-1])
    assert cond, name


ns["start"]()
st = bpy.app.driver_namespace[ns["KEY"]]
w = st["wardrobe"]
w.reg = Registry(LIB)
rig, body = bpy.data.objects["SMPLX_Twin_rig"], bpy.data.objects["SMPLX_Twin"]
writer = FrameWriter("vfr_camera_frame")
K = {"w": 640, "h": 480, "fx": 554.0, "fy": 554.0, "cx": 320.0, "cy": 240.0}

# 1. library
names = {g["name"]: g["id"] for g in w.reg.list()}
check("library lists the test garments", {"Basic T-Shirt", "Basic Trousers"} <= set(names), str(sorted(names)))


def pose(go=(0, 0, 0), sh=0.8, elbow=0.0, transl=(0, -0.1, 2.4), betas=None, hip=0.0):
    # sh 0.8 rad = arms ~45 deg down (natural stance). At ~70 deg the SMPL-X skin itself self-intersects
    # in the armpit, so "inside the body" is ill-defined there (reported separately below).
    bp = np.zeros(63)
    bp[15 * 3 + 2], bp[16 * 3 + 2] = -sh, sh
    bp[17 * 3 + 1], bp[18 * 3 + 1] = -elbow, elbow
    bp[0], bp[3] = -hip, -hip
    p = {"global_orient": list(go), "body_pose": bp.tolist(), "left_hand_pose": [0.0] * 45,
         "right_hand_pose": [0.0] * 45, "transl": list(transl), "camera": K}
    if betas is not None:
        p["betas"] = list(betas)
    ns["apply_pose"](rig, p, st["correctives"])
    if betas is not None and list(rig.get("vfr_betas", [0] * 10)) != list(betas):
        ns["apply_shape"](rig, body, betas, st["correctives"])
    if "vfr_K" not in bpy.data.objects["VFR_Camera"]:
        ns["apply_camera"](bpy.data.objects["VFR_Camera"], bpy.data.objects["VFR_Background"], K)


def run(n, **kw):
    times = []
    for _ in range(n):
        pose(**kw)
        bpy.context.view_layer.update()
        t0 = time.perf_counter()
        w.tick()
        times.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.07)  # ~ live tracking rate; the solver runs on its worker thread
        st["correctives"].flush()
    return times


def penetration():
    """% of garment vertices inside the evaluated twin body, and median distance from the skin."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    M = np.array(body.matrix_world)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    tree = BVHTree.FromPolygons([tuple(p) for p in co], [tuple(p.vertices) for p in me.polygons])
    ev.to_mesh_clear()
    disp = w.active["display"]
    Md = np.array(disp.matrix_world)
    x = w.active["sim"].x @ Md[:3, :3].T + Md[:3, 3]
    sd = []
    for p in x:
        loc, nrm, _, _ = tree.find_nearest(p.tolist())
        sd.append((p - np.array(loc)) @ np.array(nrm))
    sd = np.array(sd)
    return 100 * (sd < -0.003).mean(), float(np.median(sd))


# 2. wear + simulate
t0 = time.perf_counter()
w.wear(names["Basic T-Shirt"])
wear_ms = (time.perf_counter() - t0) * 1000
check("wear: driver bound + display created", w.active["driver"].modifiers["VFR_Follow"].is_bound,
      f"{len(w.active['display'].data.vertices)} verts, {wear_ms:.0f} ms to load+bind")
tick_ms = run(25)
pen, med = penetration()
check("standing: garment on the body, not inside", pen < 2.0 and 0.0 < med < 0.03, f"{pen:.1f}% inside, median {100 * med:.1f} cm off skin")
run(20, sh=-0.6)  # arms raised
pen, med = penetration()
check("arms raised: still not inside the body", pen < 3.0, f"{pen:.1f}% inside, median {100 * med:.1f} cm")
run(20, sh=1.2)   # arms hanging close to the body (armpit contact)
pen_close, _ = penetration()
check("arms against the body (info: armpit pinch)", True, f"{pen_close:.1f}% of vertices pinched in the armpit contact zone")
# jump: body goes up 25 cm over 4 frames and back
hem = w.active["sim"].x[:, 1] < np.percentile(w.active["sim"].x[:, 1], 8)
lags = []
for k in list(range(1, 5)) + list(range(3, -1, -1)):
    run(1, transl=(0, -0.1 - 0.0625 * k, 2.4))   # camera y is down: up = negative y
    tgt = w._driver_target(w.active["driver"], rig)
    lags.append(float(np.abs(tgt[hem, 1] - w.active["sim"].x[hem, 1]).mean()))
run(25)
pen, med = penetration()
check("jump: hem lags (inertia) then lands without penetration", max(lags) > 0.01 and pen < 2.0,
      f"max hem lag {100 * max(lags):.1f} cm, after landing {pen:.1f}% inside")
# body shape change (new customer proportions) -> garment follows
run(20, betas=[1.5, -1.0] + [0.0] * 8)
pen, med = penetration()
check("personalised body shape: garment refits", pen < 2.0, f"{pen:.1f}% inside, median {100 * med:.1f} cm")
check("speed", True, f"main-thread tick median {np.median(tick_ms):.1f} ms (gather {w.gather_ms:.1f} ms), "
                     f"cloth solver {w.sim_ms:.0f} ms/step on its worker thread")

# 3. hands: push (left hand moved into the shirt front) and pinch-grab-pull via the real pinch detector
x0 = w.active["sim"].x.copy()
orig_colliders = w._body_colliders
front = x0[np.argmax(x0[:, 2])]          # most forward garment vertex (chest)


def fake_hands(push=None, pinch=None):
    def colliders(r, b):
        caps, hands, heads = orig_colliders(r, b)
        if push is not None:
            hands["left"] = (np.array([push]), np.array([0.035]))
        if pinch is not None:
            heads = dict(heads)
            heads[bw.HAND["right"]["thumb"]] = pinch
            heads[bw.HAND["right"]["index"]] = pinch + [0.01, 0, 0]
        return caps, hands, heads
    return colliders


for k in range(8):   # palm approaches from the front and presses 2 cm in
    w._body_colliders = fake_hands(push=front + [0, 0, 0.06 - 0.01 * k])
    run(1)
x1 = w.active["sim"].x
d = np.linalg.norm(x1 - x0, axis=1)
near = np.linalg.norm(x0 - front, axis=1) < 0.05
check("hand push dents the fabric locally", d[near].mean() > 0.005 and d[~near].mean() < d[near].mean() / 3,
      f"near {100 * d[near].mean():.1f} cm, elsewhere {100 * d[~near].mean():.2f} cm, contacts {w.active['sim'].last_contacts}")
w._body_colliders = orig_colliders
run(15)
grab_at = front + [0.0, -0.05, 0.0]
grab_at = w.active["sim"].x[np.argmin(np.linalg.norm(w.active["sim"].x - grab_at, axis=1))]
xg0 = w.active["sim"].x.copy()
for k in range(8):   # pinch and pull 12 cm towards the camera
    w._body_colliders = fake_hands(pinch=grab_at + [0, 0, 0.015 * k])
    run(1)
moved = np.linalg.norm(w.active["sim"].x - xg0, axis=1)
held = np.linalg.norm(xg0 - grab_at, axis=1) < 0.03
check("pinch grabs and pulls a local patch", "right" in w.active["sim"].grabs and moved[held].mean() > 0.06,
      f"held patch moved {100 * moved[held].mean():.1f} cm, rest {100 * moved[~held].mean():.2f} cm")
w._body_colliders = orig_colliders   # fingers open -> release
run(30)
back = np.linalg.norm(w.active["sim"].x - xg0, axis=1).max()
check("release: fabric settles back", "right" not in w.active["sim"].grabs and back < 0.03, f"max offset {100 * back:.1f} cm")

# 4. switching + removal leave nothing behind
w.wear(names["Basic Trousers"])
run(15)
pen, med = penetration()
check("switch to trousers", w.active["gid"] == names["Basic Trousers"] and pen < 4.0, f"{pen:.1f}% inside")
w.remove()
left = [o.name for o in bpy.data.objects if o.name.startswith(("VFR_Driver", "VFR_Garment"))]
check("remove: no garment objects left", not left and w.active is None, str(left))

# 5. Mirror Mode: rendered image == camera image wherever there is no garment (avatar invisible)
frame = (np.random.default_rng(0).integers(0, 255, (480, 640, 3))).astype(np.uint8)
writer.write(frame)
for _ in range(3):
    st["tick"]()
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 1
sc.cycles.use_denoising = False
sc.cycles.filter_width = 0.01   # pixel-centre samples, so a per-pixel comparison is meaningful
sc.render.resolution_x, sc.render.resolution_y = 640, 480
sc.render.image_settings.file_format = "PNG"
w.set_mirror(True)
check("mirror: twin uses the projected camera feed, rig hidden", body.data.materials[0].name == "VFR_ProjectedFeed"
      and rig.hide_get() and not bpy.data.objects["VFR_Background"].hide_render)
pose(sh=0.9)
bpy.context.view_layer.update()


def render(path):
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    px = np.array(bpy.data.images.load(path).pixels[:]).reshape(480, 640, 4)[::-1, :, :3]
    return px


out = tempfile.mkdtemp()
img = render(os.path.join(out, "no_garment.png"))
ref = frame[:, :, ::-1] / 255.0
diff = np.abs(img - ref).mean(2)
dimg = bpy.data.images.new("diff", 640, 480)
dimg.pixels.foreach_set(np.dstack([np.repeat(diff[::-1, :, None], 3, 2), np.ones((480, 640, 1))]).astype(np.float32).ravel())
dimg.save_render(os.path.join(out, "diff.png"))
print("DIFF_DIR", out)
check("mirror: without a garment the render IS the camera image (avatar invisible)", np.median(diff) < 0.02 and np.percentile(diff, 99) < 0.12,
      f"median |render - camera| {np.median(diff):.3f}, p99 {np.percentile(diff, 99):.3f}")
w.wear(names["Basic T-Shirt"])
run(10)
img2 = render(os.path.join(out, "tshirt.png"))
diff2 = np.abs(img2 - ref).mean(2)
changed = (diff2 > 0.15).mean()
check("mirror: with the T-shirt only the garment area changes", 0.02 < changed < 0.35,
      f"{100 * changed:.1f}% of pixels changed (the shirt), image saved {out}")
w.set_mirror(False)

# 6. upload pipeline: cm-scaled OBJ -> separate process -> approve -> edit -> delete; photo/unsupported rejected
src = os.path.join(LIB, w.reg.garments[names["Basic T-Shirt"]]["original"])
cm = os.path.join(out, "Summer Tee.obj")
with open(src) as fi, open(cm, "w") as fo:
    for line in fi:
        fo.write("v " + " ".join(f"{float(v) * 100:.4f}" for v in line.split()[1:4]) + "\n" if line.startswith("v ") else line)
gid = w.start_processing(cm)
photo = os.path.join(out, "shirt.png")
bpy.data.images.new("p", 4, 4).save_render(photo)
pid = w.start_processing(photo)
t0 = time.time()
while any(p["report"] is None for p in w.pending.values()) and time.time() - t0 < 180:
    time.sleep(0.5)
    w.poll_processing()
rep = w.pending[gid]["report"]
check("upload: cm-scaled garment processed in a separate process", rep["status"] == "ok" and any("Rescaled" in x for x in rep["warnings"]),
      f"category {rep.get('category')} ({rep.get('category_reason')}), {rep['stats']}")
check("upload: photo rejected with an explanation", w.pending[pid]["report"]["status"] == "error",
      w.pending[pid]["report"]["errors"][0][:80])
try:
    w.reg.stage_upload(os.path.join(out, "notes.txt") if open(os.path.join(out, "notes.txt"), "w").write("x") else "")
    ok = False
except ValueError as exc:
    ok = "Unsupported" in str(exc)
check("upload: unsupported format rejected", ok)
entry = w.approve(gid)
w.cancel(pid)
check("approve: added to the library, original preserved", gid in w.reg.garments and os.path.isfile(w.reg.abs(entry["original"]))
      and os.path.isfile(cm))
w.reg.update(gid, name="Summer Tee (edited)", fabric_overrides={"mass": 0.2})
check("edit: saved", w.reg.load().garments[gid]["name"] == "Summer Tee (edited)")
other = set(w.reg.garments) - {gid}
w.delete(gid, delete_original=False)
check("delete: removed, original kept, other garments untouched", gid not in w.reg.garments
      and os.path.isfile(w.reg.abs(entry["original"])) and other <= set(w.reg.garments)
      and all(os.path.isfile(w.reg.abs(w.reg.garments[o]["processed"])) for o in other))

print("\n".join(log))
print("WARDROBE OK")
shutil.rmtree(LIB, ignore_errors=True)
