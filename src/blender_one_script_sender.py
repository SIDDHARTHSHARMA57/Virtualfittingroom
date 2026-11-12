
import os, sys, json, time, socket, cv2
PROJECT = "/media/siddharth-sharam/Workspace/VirtualFittingRoom"
sys.path.insert(0, PROJECT)
from src.tracker import MediaPipePoseTracker
MODEL = os.path.join(PROJECT, "models", "pose_landmarker_lite.task")
HOST, PORT = "127.0.0.1", 8765
tracker = MediaPipePoseTracker(model_path=MODEL, min_conf=0.5)
camera = cv2.VideoCapture(0)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam index 0")
    while True:
        try:
            client.connect((HOST, PORT))
            break
        except OSError:
            time.sleep(1)
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        result = tracker.process_frame(frame)
        people = result.get("persons", [])
        pts = []
        if people:
            for p in people[0]["keypoints"]:
                pts.append([float(p[0])/frame.shape[1],
                            float(p[1])/frame.shape[0],
                            float(p[2])])
        client.sendall((json.dumps({"points": pts})+"\n").encode())
        if pts:
            for x, y, conf in pts:
                if conf >= 0.35:
                    cv2.circle(frame, (int(x*frame.shape[1]),
                                       int(y*frame.shape[0])),
                               4, (0,255,0), -1)
        cv2.putText(frame, "Live tracking -> Blender", (12,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
        cv2.imshow("VirtualFittingRoom Camera", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
finally:
    camera.release()
    client.close()
    cv2.destroyAllWindows()
    tracker.detector.close()
