"""
Calibrate the webcam (intrinsics + lens distortion) with a printed checkerboard.

  python scripts/calibrate_camera.py --device 0 --width 640 --height 480 --cols 9 --rows 6 --square 0.025

--cols/--rows = INNER corners (a 10x7-square board has 9x6 inner corners); --square = square size in m.
Close the dashboard first (it holds the camera). Hold the board at different distances / angles /
image corners; SPACE captures a view when the corners are found (green), C calibrates (>= 12 views),
Q quits. Writes config/camera_intrinsics.json, which the tracking service and the Blender camera use.
Calibrate on the RAW (un-mirrored) frames; the service mirrors cx itself.
"""
import os
import json
import argparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "config", "camera_intrinsics.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=0.025)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.device, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {a.device} (is the dashboard still running?)")

    pattern = (a.cols, a.rows)
    obj = np.zeros((a.cols * a.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
    obj_pts, img_pts, size = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_FAST_CHECK)
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, pattern, corners, found)
        cv2.putText(view, f"views {len(obj_pts)}  SPACE=capture  C=calibrate  Q=quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("VFR camera calibration", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" ") and found:
            obj_pts.append(obj)
            img_pts.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit))
        elif k == ord("c") and len(obj_pts) >= 12:
            break
        elif k == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return
    cap.release()
    cv2.destroyAllWindows()

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    res = {"width": size[0], "height": size[1], "fx": float(K[0, 0]), "fy": float(K[1, 1]),
           "cx": float(K[0, 2]), "cy": float(K[1, 2]), "dist": dist.ravel().tolist(),
           "rms_px": float(rms), "views": len(obj_pts),
           "hfov_deg": float(np.degrees(2 * np.arctan(size[0] / (2 * K[0, 0]))))}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"Saved {OUT}  (rms {rms:.2f} px; < 0.5 px is good, > 1 px: recapture)")


if __name__ == "__main__":
    main()
