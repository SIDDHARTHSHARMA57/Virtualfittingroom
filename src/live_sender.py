import os, sys, json, time, socket, cv2

PROJECT = "/media/siddharth-sharam/Workspace/VirtualFittingRoom"
sys.path.insert(0, PROJECT)

from src.tracker import MediaPipePoseTracker

HOST, PORT = "127.0.0.1", 8766
MODEL = os.path.join(PROJECT, "models", "pose_landmarker_lite.task")

tracker = MediaPipePoseTracker(model_path=MODEL, min_conf=0.5)
camera = cv2.VideoCapture(0)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    if not camera.isOpened():
        raise RuntimeError("Cannot open camera 0. Close other camera apps.")

    print("[SENDER] Waiting for Blender receiver...")
    while True:
        try:
            client.connect((HOST, PORT))
            break
        except OSError:
            time.sleep(1)

    print("[SENDER] Connected. Press Q in camera window to quit.")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("[SENDER] Camera frame failed.")
            break

        frame = cv2.flip(frame, 1)
        result = tracker.process_frame(frame)
        people = result.get("persons", [])
        points = []

        if people:
            for p in people[0]["keypoints"]:
                points.append([
                    float(p[0]) / frame.shape[1],
                    float(p[1]) / frame.shape[0],
                    float(p[2])
                ])

        packet = json.dumps({"points": points}) + "\n"
        client.sendall(packet.encode("utf-8"))

        for p in points:
            x, y, confidence = p
            if confidence >= 0.35:
                cv2.circle(
                    frame,
                    (int(x * frame.shape[1]), int(y * frame.shape[0])),
                    4, (0, 255, 0), -1
                )

        status = "BODY DETECTED" if points else "NO BODY DETECTED"
        cv2.putText(frame, status + " | Q = quit", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("VirtualFittingRoom - Camera Sender", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

finally:
    camera.release()
    client.close()
    cv2.destroyAllWindows()
    tracker.detector.close()
    print("[SENDER] Stopped.")
