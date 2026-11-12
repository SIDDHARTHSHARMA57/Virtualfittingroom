import os
import sys
import json
import time
import socket
import cv2

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from src.tracker import MediaPipePoseTracker

HOST = "127.0.0.1"
PORT = 8765
MODEL = os.path.join(PROJECT, "models", "pose_landmarker_lite.task")

tracker = MediaPipePoseTracker(model_path=MODEL, min_conf=0.5)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    tracker.detector.close()
    raise RuntimeError("Could not open camera 0")

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.settimeout(1.0)

print("Waiting for Blender's receiver on 127.0.0.1:8765 ...")

try:
    while True:
        try:
            client.connect((HOST, PORT))
            break
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(1)

    client.settimeout(None)
    print("Connected to Blender. Press Q to stop.")

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        result = tracker.process_frame(frame)
        persons = result.get("persons", [])

        points = []
        if persons:
            for p in persons[0]["keypoints"]:
                points.append([
                    float(p[0]) / frame.shape[1],
                    float(p[1]) / frame.shape[0],
                    float(p[2])
                ])

        packet = json.dumps({"points": points}) + "\n"
        client.sendall(packet.encode("utf-8"))

        if points:
            for x, y, confidence in points:
                if confidence >= 0.35:
                    cv2.circle(
                        frame,
                        (int(x * frame.shape[1]), int(y * frame.shape[0])),
                        4, (0, 255, 0), -1
                    )

        cv2.putText(
            frame, "Sending landmarks to Blender",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 0), 2
        )
        cv2.imshow("VirtualFittingRoom - Blender Sender", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

except (BrokenPipeError, ConnectionResetError) as e:
    print("Blender connection closed:", e)
finally:
    camera.release()
    client.close()
    cv2.destroyAllWindows()
    tracker.detector.close()
    print("Sender stopped.")
