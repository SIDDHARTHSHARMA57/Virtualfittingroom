"""
Service Entry Point for VirtualFittingRoom Camera & Tracking Subsystem.
Parses CLI flags or config YAML, initializes components, and manages lifecycle.
"""
import os
import sys
import signal
import logging
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.camera.capture import ThreadedCamera
from src.camera.tracking import UnifiedTracker
from src.camera.pipeline import TrackingPipeline

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )

def main():
    parser = argparse.ArgumentParser(description="VirtualFittingRoom Camera & Tracking Service")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--fourcc", type=str, default="MJPG", help="FourCC code (default: MJPG)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Blender host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8766, help="Blender port (default: 8766)")
    parser.add_argument("--preview", action="store_true", help="Enable OpenCV preview window")
    parser.add_argument("--no-hands", action="store_true", help="Disable hand tracking for performance")
    parser.add_argument("--no-face", action="store_true", help="Disable face tracking for performance")
    parser.add_argument("--no-smooth", action="store_true", help="Disable 1-Euro temporal smoothing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("VFR-Main")
    logger.info("Initializing VirtualFittingRoom Tracking Service...")

    pose_model = os.path.join(PROJECT_ROOT, "models/pose_landmarker_lite.task")
    hand_model = os.path.join(PROJECT_ROOT, "models/hand_landmarker.task")
    face_model = os.path.join(PROJECT_ROOT, "models/face_landmarker.task")

    camera = ThreadedCamera(
        device_id=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        stale_threshold_s=0.5
    )

    tracker = UnifiedTracker(
        pose_model_path=pose_model,
        hand_model_path=hand_model,
        face_model_path=face_model,
        enable_pose=True,
        enable_hands=not args.no_hands,
        enable_face=not args.no_face,
        enable_smoothing=not args.no_smooth,
    )

    pipeline = TrackingPipeline(
        camera=camera,
        tracker=tracker,
        host=args.host,
        port=args.port,
        show_preview=args.preview,
    )

    # Signal handling for graceful termination
    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, terminating pipeline cleanly...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    pipeline.start()

if __name__ == "__main__":
    main()
