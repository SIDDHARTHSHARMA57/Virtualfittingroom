import bpy, socket, threading, json, time, subprocess, os
from mathutils import Quaternion

PROJECT = "/media/siddharth-sharam/Workspace/VirtualFittingRoom"
PYTHON = os.path.join(PROJECT, "envs", "smplx", "bin", "python")
HOST, PORT = "127.0.0.1", 8766

# State shared with the receiver thread
state = {"points": None, "status": "Starting receiver"}
lock = threading.Lock()
stop_event = threading.Event()

# Remove only objects created by this live skeleton script
for name in ("VFR_Live_Skeleton", "VFR_Live_Lines", "VFR_Live_Status"):
    obj = bpy.data.objects.get(name)
    if obj:
        bpy.data.objects.remove(obj, do_unlink=True)

EDGES = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),
    (11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]

mesh = bpy.data.meshes.new("VFR_LandmarkMesh")
mesh.from_pydata([(0,0,0)] * 33, EDGES, [])
mesh.update()
skeleton = bpy.data.objects.new("VFR_Live_Skeleton", mesh)
bpy.context.collection.objects.link(skeleton)
skeleton.show_in_front = True
skeleton.display_type = 'WIRE'

curve = bpy.data.curves.new("VFR_Live_Curves", "CURVE")
curve.dimensions = "3D"
curve.bevel_depth = 0.018
curve.bevel_resolution = 2
for a, b in EDGES:
    sp = curve.splines.new("POLY")
    sp.points.add(1)
    sp.points[0].co = (0,0,0,1)
    sp.points[1].co = (0,0,0,1)

lines = bpy.data.objects.new("VFR_Live_Lines", curve)
bpy.context.collection.objects.link(lines)
lines.show_in_front = True

text_data = bpy.data.curves.new("VFR_Status_Font", "FONT")
text_data.body = "VFR: Starting receiver..."
text_data.size = 0.16
status_obj = bpy.data.objects.new("VFR_Live_Status", text_data)
bpy.context.collection.objects.link(status_obj)
status_obj.location = (-2, 0, 2.3)
status_obj.show_in_front = True

def receiver():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((HOST, PORT))
        server.listen(1)
        server.settimeout(1)
        with lock:
            state["status"] = "Waiting for camera sender..."
        while not stop_event.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with lock:
                state["status"] = "Camera connected; waiting for landmarks..."
            with conn:
                conn.settimeout(1)
                buffer = b""
                while not stop_event.is_set():
                    try:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        buffer += chunk
                        while b"\n" in buffer:
                            raw, buffer = buffer.split(b"\n", 1)
                            try:
                                data = json.loads(raw.decode("utf-8"))
                                pts = data.get("points", [])
                                with lock:
                                    if len(pts) == 33:
                                        state["points"] = pts
                                        state["status"] = "LIVE: 33 landmarks received"
                                    else:
                                        state["status"] = "Connected; stand in camera view"
                            except (ValueError, UnicodeDecodeError):
                                pass
            with lock:
                state["status"] = "Sender disconnected; waiting..."
    except Exception as exc:
        with lock:
            state["status"] = "Receiver error: " + str(exc)
    finally:
        server.close()

threading.Thread(target=receiver, daemon=True).start()

def update_skeleton():
    if not bpy.data.objects.get("VFR_Live_Skeleton"):
        return None

    with lock:
        pts = state["points"]
        status = state["status"]

    status_obj.data.body = status[:80]

    if pts and len(pts) == 33:
        coords = [
            ((0.5-float(p[0]))*4.0, 0.0,
             (0.5-float(p[1]))*4.0)
            for p in pts
        ]

        for i, co in enumerate(coords):
            skeleton.data.vertices[i].co = co
        skeleton.data.update()

        for i, (a, b) in enumerate(EDGES):
            curve.splines[i].points[0].co = (*coords[a], 1.0)
            curve.splines[i].points[1].co = (*coords[b], 1.0)
        curve.update_tag()

    return 1.0 / 30.0

bpy.app.timers.register(update_skeleton, first_interval=0.2)

# Start the camera sender as a child process of Blender
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
sender_log = open(os.path.join(PROJECT, "logs", "live_sender.log"), "w")
sender_process = subprocess.Popen(
    [PYTHON, os.path.join(PROJECT, "src", "live_sender.py")],
    cwd=PROJECT, env=env,
    stdout=sender_log, stderr=subprocess.STDOUT
)

# Frame the live skeleton area
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            region = area.spaces.active.region_3d
            region.view_perspective = "ORTHO"
            region.view_rotation = Quaternion((1,0,0,0))
            region.view_location = (0,0,0)
            region.view_distance = 5

print("[VFR] Receiver and camera sender launched.")
