import os
import sys
import time
import yaml
import logging
import cv2
import torch
from collections import deque

from camera import CameraManager
from tracker import create_tracker, BasePoseTracker
from visualizer import PoseVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VirtualFittingRoom")

def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, config_path)
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_app(headless_frames: int = 0):
    """
    Main live pose tracking application loop.
    If headless_frames > 0, runs without GUI window for benchmarking/automated validation.
    """
    config = load_config()
    cam_cfg = config.get("camera", {})
    tracker_cfg = config.get("pose_tracker", {})
    disp_cfg = config.get("display", {})

    logger.info("Starting Milestone 1 - Live Pose Tracking Application...")

    # Hardware detection for display HUD
    device_name = "CPU"
    if tracker_cfg.get("device", "").startswith("cuda") and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    # 1. Initialize Camera
    camera = CameraManager(
        device_index=cam_cfg.get("device_index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        target_fps=cam_cfg.get("target_fps", 30),
        buffer_size=cam_cfg.get("buffer_size", 1)
    )

    if not camera.start():
        logger.error("Could not start camera. Exiting...")
        return

    # 2. Initialize Tracker
    current_backend = tracker_cfg.get("backend", "yolo")
    tracker = create_tracker(tracker_cfg)

    # 3. Initialize Visualizer
    visualizer = PoseVisualizer(disp_cfg)
    window_name = disp_cfg.get("window_name", "VirtualFittingRoom")
    mirror = disp_cfg.get("mirror_display", True)

    if headless_frames == 0:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Rolling FPS measurement window
    fps_history = deque(maxlen=30)
    frame_count = 0
    t_start = time.perf_counter()

    try:
        while True:
            t_frame_start = time.perf_counter()
            ret, frame = camera.read()
            if not ret or frame is None:
                logger.warning("Camera frame read failed. Retrying in 100ms...")
                time.sleep(0.1)
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            # Pose inference
            tracking_data = tracker.process_frame(frame)

            # Measure overall frame processing time
            t_frame_end = time.perf_counter()
            frame_duration = t_frame_end - t_frame_start
            fps_history.append(frame_duration)
            avg_fps = len(fps_history) / sum(fps_history) if sum(fps_history) > 0 else 0.0

            # Render UI
            annotated_frame = visualizer.render(frame, avg_fps, tracking_data, device_name)

            frame_count += 1

            if headless_frames > 0:
                if frame_count % 10 == 0 or frame_count == headless_frames:
                    logger.info(f"[Benchmark Frame {frame_count}/{headless_frames}] Measured FPS: {avg_fps:.2f} | Latency: {tracking_data['latency_ms']:.1f}ms | Persons: {len(tracking_data['persons'])}")
                if frame_count >= headless_frames:
                    logger.info("Headless benchmark completed successfully.")
                    break
            else:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # ESC or Q
                    logger.info("User requested exit.")
                    break
                elif key == ord('m'): # Toggle mirror
                    mirror = not mirror
                    logger.info(f"Mirror mode toggled: {mirror}")
                elif key == ord('b'): # Switch backend dynamically
                    new_backend = "mediapipe" if current_backend == "yolo" else "yolo"
                    logger.info(f"Switching backend from {current_backend} to {new_backend}...")
                    tracker_cfg["backend"] = new_backend
                    try:
                        tracker = create_tracker(tracker_cfg)
                        current_backend = new_backend
                    except Exception as e:
                        logger.error(f"Failed to switch backend: {e}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        total_time = time.perf_counter() - t_start
        overall_fps = frame_count / total_time if total_time > 0 else 0.0
        logger.info(f"Session finished. Total frames: {frame_count}, Elapsed: {total_time:.2f}s, Overall Average FPS: {overall_fps:.2f}")
        camera.release()
        if headless_frames == 0:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    headless = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_app(headless_frames=headless)
