import bpy, os, subprocess, traceback

PROJECT = "/media/siddharth-sharam/Workspace/VirtualFittingRoom"
PYTHON = PROJECT + "/envs/smplx/bin/python"
CAMERA = PROJECT + "/src/camera/main.py"
RECEIVER = PROJECT + "/src/blender_pose_receiver.py"
CONTROLLER = PROJECT + "/src/blender_arms_head.py"

def run_script(path):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    exec(compile(source, path, "exec"), {"__name__": "__main__"})

try:
    if not bpy.data.objects.get("Human.rig"):
        raise RuntimeError("Armature Human.rig not found in Blender scene")
    if not bpy.data.objects.get("Human"):
        raise RuntimeError("Mesh Human not found in Blender scene")
    if not os.path.isfile(PYTHON):
        raise RuntimeError("Python environment not found: " + PYTHON)

    print("[VFR] Starting Blender pose receiver...")
    run_script(RECEIVER)

    print("[VFR] Starting arm and head controller...")
    run_script(CONTROLLER)

    print("[VFR] Starting camera tracker...")
    log_path = PROJECT + "/logs/camera-launch.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", buffering=1)
    subprocess.Popen(
        [PYTHON, CAMERA, "--preview"],
        cwd=PROJECT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    print("[VFR] Startup commands sent. Check the camera preview and Blender console.")

except Exception:
    print("[VFR] STARTUP FAILED:")
    traceback.print_exc()
