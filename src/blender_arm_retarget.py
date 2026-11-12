import bpy
import math

RECEIVER_KEY = "VFR_POSE_RECEIVER"
rig = bpy.data.objects.get("Human.rig")
state = bpy.app.driver_namespace.get(RECEIVER_KEY)

if rig is None or rig.type != "ARMATURE":
    raise RuntimeError("Human.rig was not found")
if state is None:
    raise RuntimeError("Blender pose receiver is not running")

# Stop a previous diagnostic timer, if any.
bpy.app.driver_namespace["VFR_ARM_RETARGET_ENABLED"] = False

def arm_update():
    if not bpy.app.driver_namespace.get("VFR_ARM_RETARGET_ENABLED"):
        return None

    s = bpy.app.driver_namespace.get(RECEIVER_KEY)
    points = s.get("last_points") if s else None

    if points and len(points) == 33:
        for side, si, ei, wi in (
            ("left", 11, 13, 15),
            ("right", 12, 14, 16),
        ):
            sh, el, wr = points[si], points[ei], points[wi]
            if min(sh[2], el[2], wr[2]) >= 0.35:
                upper = math.degrees(math.atan2(
                    -(el[1] - sh[1]), el[0] - sh[0]
                ))
                lower = math.degrees(math.atan2(
                    -(wr[1] - el[1]), wr[0] - el[0]
                ))
                s["debug_" + side + "_arm_angles"] = (
                    round(upper, 1), round(lower, 1)
                )
    return 0.03

bpy.app.driver_namespace["VFR_ARM_RETARGET_ENABLED"] = True
bpy.app.timers.register(arm_update, first_interval=0.03)
print("[VFR] Arm-angle diagnostic running. No bones are changed.")
