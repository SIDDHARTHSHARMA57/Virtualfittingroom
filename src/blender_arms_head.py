import bpy
import math
from mathutils import Vector, Euler

KEY = "VFR_ARMS_HEAD"
RECEIVER_KEY = "VFR_POSE_RECEIVER"
RIG_NAME = "Human.rig"
MESH_NAME = "Human"

# MediaPipe Pose indices
SIDES = {
    "L": {"shoulder": 11, "elbow": 13, "wrist": 15},
    "R": {"shoulder": 12, "elbow": 14, "wrist": 16},
}

HEAD_IDS = {
    "nose": 0, "left_eye": 2, "right_eye": 5,
    "left_ear": 7, "right_ear": 8,
}

# Stop a previous run of this script, if any.
old = bpy.app.driver_namespace.get(KEY)
if old:
    old["active"] = False
    for obj in old.get("targets", []):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass
    for bone_name, constraint_name in old.get("constraints", []):
        rig_old = bpy.data.objects.get(RIG_NAME)
        pb = rig_old.pose.bones.get(bone_name) if rig_old else None
        c = pb.constraints.get(constraint_name) if pb else None
        if c:
            pb.constraints.remove(c)
    rig_old = bpy.data.objects.get(RIG_NAME)
    if rig_old:
        for name, rot in old.get("original_rotations", {}).items():
            pb = rig_old.pose.bones.get(name)
            if pb:
                pb.rotation_mode = "XYZ"
                pb.rotation_euler = rot
    bpy.app.driver_namespace.pop(KEY, None)

rig = bpy.data.objects.get(RIG_NAME)
mesh = bpy.data.objects.get(MESH_NAME)
receiver = bpy.app.driver_namespace.get(RECEIVER_KEY)

if not rig or rig.type != "ARMATURE":
    raise RuntimeError("Could not find armature Human.rig")
if not mesh:
    raise RuntimeError("Could not find mesh Human")
if not receiver:
    raise RuntimeError("Pose receiver is not running. Start the receiver first.")

required = [
    "shoulder01.L", "upperarm01.L", "upperarm02.L",
    "lowerarm01.L", "lowerarm02.L", "wrist.L",
    "shoulder01.R", "upperarm01.R", "upperarm02.R",
    "lowerarm01.R", "lowerarm02.R", "wrist.R", "head"
]
missing = [n for n in required if n not in rig.pose.bones]
if missing:
    raise RuntimeError("Required bones missing: " + ", ".join(missing))

# Keep original head rotation for safe restoration.
original_rotations = {}
head = rig.pose.bones.get("head")
if head:
    head.rotation_mode = "XYZ"
    original_rotations["head"] = head.rotation_euler.copy()

# The existing arm chains are multi-segment chains. IK is placed on
# lowerarm02 and solves back through the forearm and upper-arm segments.
targets = []
constraints = []

def make_target(name):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.empty_display_type = "SPHERE"
    obj.empty_display_size = 0.025
    obj.hide_render = True
    obj.show_in_front = True
    targets.append(obj)
    return obj

for side in ("L", "R"):
    pb = rig.pose.bones["lowerarm02." + side]
    cname = "VFR_ArmIK_" + side
    existing = pb.constraints.get(cname)
    if existing:
        pb.constraints.remove(existing)

    target = make_target("VFR_Target_Wrist_" + side)
    pole = make_target("VFR_Target_Elbow_" + side)

    con = pb.constraints.new("IK")
    con.name = cname
    con.target = target
    con.pole_target = pole
    con.chain_count = 4
    con.use_stretch = False
    con.influence = 0.0
    constraints.append(("lowerarm02." + side, cname))

# Coordinate calibration from rest shoulder span.
bl = rig.data.bones["shoulder01.L"].head_local
br = rig.data.bones["shoulder01.R"].head_local
rest_mid = (bl + br) * 0.5
rest_span = abs(bl.x - br.x)
if rest_span < 0.01:
    raise RuntimeError("Unexpected shoulder span in rig")

state = {
    "active": True,
    "rig": rig,
    "targets": targets,
    "constraints": constraints,
    "original_rotations": original_rotations,
    "rest_mid": rest_mid.copy(),
    "rest_span": rest_span,
    "smoothed": {},
    "last_head": None,
}
bpy.app.driver_namespace[KEY] = state

def lerp(a, b, t):
    return a + (b - a) * t

def point3(x, y, center_x, center_y, scale, depth):
    # Camera X maps to avatar X; image Y maps to avatar Z.
    return Vector((
        rest_mid.x + (x - center_x) * scale,
        depth,
        rest_mid.z + (center_y - y) * scale,
    ))

def animate():
    st = bpy.app.driver_namespace.get(KEY)
    if not st or not st.get("active"):
        return None

    s = bpy.app.driver_namespace.get(RECEIVER_KEY)
    pts = s.get("last_points") if s else None
    if not pts or len(pts) != 33:
        return 0.03

    # Reject landmarks that are outside the image or low confidence.
    def valid(i, threshold=0.55):
        x, y, c = map(float, pts[i][:3])
        return (
            math.isfinite(x) and math.isfinite(y) and math.isfinite(c)
            and c >= threshold
            and -0.25 <= x <= 1.25
            and -0.25 <= y <= 1.75
        )

    # Require both shoulders for a stable image-to-avatar calibration.
    if not valid(11, 0.55) or not valid(12, 0.55):
        return 0.03

    ls = pts[11]
    rs = pts[12]
    cx = (float(ls[0]) + float(rs[0])) * 0.5
    cy = (float(ls[1]) + float(rs[1])) * 0.5
    image_span = abs(float(ls[0]) - float(rs[0]))
    if image_span < 0.08:
        return 0.03

    scale = min(st["rest_span"] / image_span, 2.0)
    scale = max(scale, 0.15)

    # Depth is held at the shoulder plane in this initial 2D pass.
    depth = rest_mid.y

    for side, ids in SIDES.items():
        # Keep the arm responsive when the wrist is briefly occluded.
        if not (valid(ids["shoulder"], 0.45) and valid(ids["elbow"], 0.35)):
            con = rig.pose.bones["lowerarm02." + side].constraints.get("VFR_ArmIK_" + side)
            if con:
                con.influence = 0.0
            continue

        sh = pts[ids["shoulder"]]
        el = pts[ids["elbow"]]
        wr = pts[ids["wrist"]]

        sx, sy = float(sh[0]), float(sh[1])
        ex, ey = float(el[0]), float(el[1])
        wx, wy, wc = float(wr[0]), float(wr[1]), float(wr[2])

        elbow = point3(ex, ey, cx, cy, scale, depth)

        # If the wrist is reliable and in-frame, use it directly.
        # Otherwise extrapolate from shoulder -> elbow as a temporary target.
        wrist_ok = (
            wc >= 0.45 and
            0.0 <= wx <= 1.0 and
            0.0 <= wy <= 1.0
        )
        if wrist_ok:
            wrist = point3(wx, wy, cx, cy, scale, depth)
        else:
            dx, dy = ex - sx, ey - sy
            length = math.hypot(dx, dy)
            if length < 0.015:
                con = rig.pose.bones["lowerarm02." + side].constraints.get("VFR_ArmIK_" + side)
                if con:
                    con.influence = 0.0
                continue
            # Approximate forearm length in image space.
            wrist = point3(
                ex + dx / length * length * 0.75,
                ey + dy / length * length * 0.75,
                cx, cy, scale, depth
            )

        # Clamp target coordinates to a safe region around the shoulder plane.
        for pos in (elbow, wrist):
            pos.x = max(rest_mid.x - st["rest_span"] * 1.5,
                        min(rest_mid.x + st["rest_span"] * 1.5, pos.x))
            pos.z = max(rest_mid.z - st["rest_span"] * 2.0,
                        min(rest_mid.z + st["rest_span"] * 2.0, pos.z))

        # Smooth target positions to reduce jitter.
        for label, pos in (("elbow_" + side, elbow), ("wrist_" + side, wrist)):
            prev = st["smoothed"].get(label)
            st["smoothed"][label] = pos if prev is None else prev.lerp(pos, 0.28)

        elbow = st["smoothed"]["elbow_" + side]
        wrist = st["smoothed"]["wrist_" + side]

        bpy.data.objects["VFR_Target_Elbow_" + side + ".001"].location = rig.matrix_world @ elbow
        bpy.data.objects["VFR_Target_Wrist_" + side + ".001"].location = rig.matrix_world @ wrist

        con = rig.pose.bones["lowerarm02." + side].constraints.get("VFR_ArmIK_" + side)
        if con:
            con.influence = 1.0

    # Conservative head tracking: estimate roll from the eye line.
    # Keep rotations modest; this is 2D pose-based, not full 3D head capture.
    if valid(2, 0.45) and valid(5, 0.45):
        le, re = pts[2], pts[5]
        dx = float(re[0]) - float(le[0])
        dy = float(re[1]) - float(le[1])
        if abs(dx) > 0.01:
            roll = math.atan2(dy, dx)
            # Small local Z rotation only; clamp to avoid extreme jumps.
            roll = max(-0.35, min(0.35, roll))
            head = rig.pose.bones["head"]
            head.rotation_mode = "XYZ"
            head.rotation_euler.z = st["original_rotations"]["head"].z + roll * 0.5

    return 0.03

bpy.app.driver_namespace[KEY]["timer"] = animate
bpy.app.timers.register(animate, first_interval=0.03)

print("[VFR] Arms + head retargeter started.")
print("[VFR] Arms use IK; low-confidence/out-of-frame joints are ignored.")
print("[VFR] Head uses a conservative eye-line roll estimate.")
print("[VFR] Stop with: bpy.app.driver_namespace['VFR_ARMS_HEAD']['active'] = False")
