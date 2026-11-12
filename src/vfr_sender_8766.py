import os
import sys
import json
import time
import socket
import cv2
import mediapipe as mp

PROJECT = "/media/siddharth-sharam/Workspace/VirtualFittingRoom"
sys.path.insert(0, PROJECT)

from src.tracker import MediaPipePoseTracker
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

POSE_MODEL = PROJECT + "/models/pose_landmarker_lite.task"
HAND_MODEL = PROJECT + "/models/hand_landmarker.task"
FACE_MODEL = PROJECT + "/models/face_landmarker.task"
HOST, PORT = "127.0.0.1", 8766

for path in (POSE_MODEL, HAND_MODEL, FACE_MODEL):
    if not os.path.isfile(path):
        raise FileNotFoundError("Missing model: " + path)

pose = MediaPipePoseTracker(model_path=POSE_MODEL, min_conf=0.5)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.45,
    min_hand_presence_confidence=0.45,
    min_tracking_confidence=0.45,
)
hands_task = vision.HandLandmarker.create_from_options(hand_options)

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.45,
    min_face_presence_confidence=0.45,
    min_tracking_confidence=0.45,
    output_facial_transformation_matrixes=True,
    output_face_blendshapes=True,
)
face_task = vision.FaceLandmarker.create_from_options(face_options)

def open_camera():
    # Prefer MJPG at 720p/30 FPS; fall back to the camera default if needed.
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(0.25)
    return camera

cam = open_camera()
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def as_float(v):
    return float(v)

try:
    if not cam.isOpened():
        raise RuntimeError("Camera 0 could not open")

    while True:
        try:
            sock.connect((HOST, PORT))
            break
        except OSError:
            time.sleep(0.5)

    print("[VFR] Expanded sender connected. Press q or ESC to quit.")

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            print("[VFR] Camera frame unavailable; attempting recovery.")
            cam.release()
            time.sleep(1.0)
            cam = open_camera()
            if not cam.isOpened():
                print("[VFR] Camera reopen failed; retrying in 2 seconds.")
                time.sleep(2.0)
            continue

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Keep the existing 33-point body tracker and its coordinate format.
        pose_result = pose.process_frame(frame)
        persons = pose_result.get("persons", [])
        body_points = []
        world_points = []

        if persons:
            person = persons[0]
            for p in person["keypoints"]:
                body_points.append([
                    as_float(p[0]) / width,
                    as_float(p[1]) / height,
                    as_float(p[2]),
                ])

            wp = person.get("world_keypoints")
            if wp is not None:
                world_points = [
                    [as_float(v) for v in row[:4]]
                    for row in wp
                ]

        # Detailed hand landmarks, grouped by handedness.
        hand_result = hands_task.detect(mp_image)
        hand_data = {}

        for idx, landmarks in enumerate(hand_result.hand_landmarks or []):
            category = "Unknown"
            if idx < len(hand_result.handedness):
                categories = hand_result.handedness[idx]
                if categories:
                    category = categories[0].category_name

            hand_data[category] = [
                [as_float(p.x), as_float(p.y), as_float(p.z)]
                for p in landmarks
            ]

        # Dense face mesh: normally 478 points with this model.
        face_result = face_task.detect(mp_image)
        face_points = []
        face_matrix = None
        blendshapes = {}

        if face_result.face_landmarks:
            face_points = [
                [as_float(p.x), as_float(p.y), as_float(p.z)]
                for p in face_result.face_landmarks[0]
            ]

        if face_result.facial_transformation_matrixes:
            matrix = face_result.facial_transformation_matrixes[0]
            face_matrix = matrix.tolist()

        if face_result.face_blendshapes:
            for item in face_result.face_blendshapes[0]:
                blendshapes[item.category_name] = as_float(item.score)

        packet = {
            "schema": 2,
            "timestamp": time.time(),
            "points": body_points,
            "world_points": world_points,
            "hands": hand_data,
            "face": face_points,
            "face_matrix": face_matrix,
            "blendshapes": blendshapes,
        }

        sock.sendall((json.dumps(packet, separators=(",", ":")) + "\n").encode())

        # Draw body landmarks for camera preview.
        for x, y, confidence in body_points:
            if confidence >= 0.35:
                px, py = int(x * width), int(y * height)
                if 0 <= px < width and 0 <= py < height:
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        cv2.putText(
            frame,
            "Body: %d | Face: %d | Hands: %d" %
            (len(body_points), len(face_points), len(hand_data)),
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
        )
        cv2.imshow("VirtualFittingRoom Expanded Tracker", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

finally:
    cam.release()
    sock.close()
    cv2.destroyAllWindows()
    for task in (hands_task, face_task):
        try:
            task.close()
        except Exception:
            pass
    try:
        pose.detector.close()
    except Exception:
        pass
    print("[VFR] Sender stopped.")
