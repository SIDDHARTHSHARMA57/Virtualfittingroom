import os
import sys
import time
import cv2

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from src.tracker import MediaPipePoseTracker

MODEL = os.path.join(PROJECT, "models", "pose_landmarker_lite.task")

if not os.path.isfile(MODEL):
    raise FileNotFoundError(f"MediaPipe model not found: {MODEL}")

tracker = MediaPipePoseTracker(model_path=MODEL, min_conf=0.5)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open camera 0.")

connections = tracker.SKELETON_CONNECTIONS
last_report = time.time()
frame_count = 0
print("Live tracking started. Move your arms and legs.")
print("Press Q or ESC in the camera window to quit.")

try:
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Camera frame read failed.")
            break

        frame = cv2.flip(frame, 1)
        result = tracker.process_frame(frame)
        persons = result["persons"]

        if persons:
            person = persons[0]
            points = person["keypoints"]
            visible = []

            for i, point in enumerate(points):
                x, y, confidence = point
                if confidence >= 0.35:
                    visible.append(i)
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            for a, b in connections:
                if a in visible and b in visible:
                    p1 = tuple(points[a][:2].astype(int))
                    p2 = tuple(points[b][:2].astype(int))
                    cv2.line(frame, p1, p2, (255, 180, 0), 2)

            status = f"BODY DETECTED | landmarks: {len(visible)}/33"
        else:
            status = "NO BODY DETECTED - stand where camera can see you"

        cv2.putText(frame, status, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracker latency: {result['latency_ms']:.1f} ms",
                    (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)

        cv2.imshow("VirtualFittingRoom - Live Pose Test", frame)

        frame_count += 1
        now = time.time()
        if now - last_report >= 5:
            print(f"Tracking: {status}; "
                  f"latency={result['latency_ms']:.1f} ms")
            last_report = now

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    tracker.detector.close()
    print("Camera released. Test finished.")
