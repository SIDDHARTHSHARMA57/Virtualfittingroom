"""
VirtualFittingRoom Blender startup hook.
Runs the existing receiver and existing arm/head retargeter in Blender.
Diagnostics are written to outputs/logs/blender_live_tracking.log.
"""
import bpy
import os
import traceback

PROJECT_ROOT = "/media/siddharth-sharam/Workspace/VirtualFittingRoom"
LOG_PATH = os.path.join(PROJECT_ROOT, "outputs", "logs", "blender_live_tracking.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log(message):
    line = "[VFR Startup] " + str(message)
    print(line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\\n")
    except Exception:
        pass

def run_existing_script(relative_path):
    script_path = os.path.join(PROJECT_ROOT, relative_path)
    if not os.path.isfile(script_path):
        raise FileNotFoundError("Required script not found: " + script_path)
    namespace = {"__name__": "__main__", "__file__": script_path}
    with open(script_path, "r", encoding="utf-8") as f:
        code = compile(f.read(), script_path, "exec")
    exec(code, namespace, namespace)
    log("Started " + relative_path)

try:
    log("Scene loaded: " + bpy.data.filepath)
    run_existing_script("src/blender_pose_receiver.py")
    run_existing_script("src/blender_arms_head.py")
    log("Receiver and arm/head retargeter initialized.")
except Exception:
    log("INITIALIZATION FAILED:")
    log(traceback.format_exc())
    raise
