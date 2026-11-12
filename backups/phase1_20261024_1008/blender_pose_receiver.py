import bpy
import socket
import json
import time

HOST, PORT = "127.0.0.1", 8766
RIG_NAME = "Human.rig"
MESH_NAME = "Human"
KEY = "VFR_POSE_RECEIVER"

old = bpy.app.driver_namespace.get(KEY)
if old:
    old["active"] = False
    for sock_key in ("client", "server"):
        try:
            old[sock_key].close()
        except Exception:
            pass

rig = bpy.data.objects.get(RIG_NAME)
human = bpy.data.objects.get(MESH_NAME)
if rig is None or rig.type != "ARMATURE":
    raise RuntimeError("Human.rig armature not found")
if human is None:
    raise RuntimeError("Human mesh not found")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
server.setblocking(False)

state = {
    "active": True,
    "server": server,
    "client": None,
    "buffer": b"",
    "last_points": None,
    "world_points": None,
    "hands": {},
    "face": [],
    "face_matrix": None,
    "blendshapes": {},
    "timestamp": None,
    "packets": 0,
    "last_error": None,
    "timer": None,
}
bpy.app.driver_namespace[KEY] = state

def receive_pose():
    s = bpy.app.driver_namespace.get(KEY)
    if s is None or not s.get("active"):
        return None

    if s["client"] is None:
        try:
            client, address = s["server"].accept()
            client.setblocking(False)
            s["client"] = client
            print("[VFR] Expanded sender connected:", address)
        except BlockingIOError:
            return 0.02
        except OSError as exc:
            s["last_error"] = str(exc)
            return 0.2

    try:
        data = s["client"].recv(262144)
        if not data:
            s["client"].close()
            s["client"] = None
            s["buffer"] = b""
            print("[VFR] Sender disconnected.")
            return 0.1
        s["buffer"] += data
    except BlockingIOError:
        pass
    except OSError as exc:
        s["last_error"] = str(exc)
        try:
            s["client"].close()
        except Exception:
            pass
        s["client"] = None
        s["buffer"] = b""
        return 0.1

    # Guard against a malformed or oversized unterminated packet.
    if len(s["buffer"]) > 8 * 1024 * 1024:
        s["buffer"] = b""
        s["last_error"] = "Discarded oversized packet buffer"

    while b"\n" in s["buffer"]:
        line, s["buffer"] = s["buffer"].split(b"\n", 1)
        try:
            packet = json.loads(line.decode("utf-8"))
            points = packet.get("points", [])
            if len(points) == 33:
                s["last_points"] = points
                s["world_points"] = packet.get("world_points", [])
                s["hands"] = packet.get("hands", {})
                s["face"] = packet.get("face", [])
                s["face_matrix"] = packet.get("face_matrix")
                s["blendshapes"] = packet.get("blendshapes", {})
                s["timestamp"] = packet.get("timestamp", time.time())
                s["packets"] += 1
                s["last_error"] = None
        except (ValueError, UnicodeDecodeError, TypeError) as exc:
            s["last_error"] = str(exc)

    return 0.02

state["timer"] = receive_pose
bpy.app.timers.register(receive_pose, first_interval=0.02)

print("[VFR] Expanded receiver listening on 127.0.0.1:8766")
print("[VFR] Body: 33 points | Hands: up to 42 points | Face: up to 478 points")
print("[VFR] Stored in VFR_POSE_RECEIVER: last_points, world_points, hands, face, face_matrix, blendshapes")
