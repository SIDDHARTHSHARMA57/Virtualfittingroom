import os
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class BasePoseTracker:
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError

class YOLOPoseTracker(BasePoseTracker):
    """
    Ultralytics YOLO Pose Estimation running with PyTorch CUDA acceleration.
    Keypoints: COCO 17 landmarks (Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles).
    """
    # COCO 17 Connections
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4), # Head
        (5, 6), # Shoulders
        (5, 7), (7, 9), # Left arm
        (6, 8), (8, 10), # Right arm
        (5, 11), (6, 12), # Torso sides
        (11, 12), # Hips
        (11, 13), (13, 15), # Left leg
        (12, 14), (14, 16) # Right leg
    ]

    def __init__(self, model_path: str, device: str = "cuda:0", conf_thresh: float = 0.5, imgsz: int = 640):
        from ultralytics import YOLO
        logger.info(f"Initializing YOLO Pose Tracker with model '{model_path}' on device '{device}'...")
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, model_path)
        
        self.model = YOLO(model_path)
        self.device = device
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        logger.info("YOLO Pose Tracker ready.")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        t0 = time.perf_counter()
        results = self.model(
            frame, 
            device=self.device, 
            conf=self.conf_thresh, 
            imgsz=self.imgsz, 
            verbose=False
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        tracked_persons = []
        if len(results) > 0 and results[0].keypoints is not None:
            r = results[0]
            # Keypoints are shape [N, 17, 3] (x, y, conf)
            kpts_data = r.keypoints.data.cpu().numpy()
            boxes_data = r.boxes.data.cpu().numpy() if r.boxes is not None else []
            
            for i, person_kpts in enumerate(kpts_data):
                box = boxes_data[i][:4] if i < len(boxes_data) else None
                box_conf = float(boxes_data[i][4]) if i < len(boxes_data) and len(boxes_data[i]) > 4 else 1.0
                tracked_persons.append({
                    "id": i,
                    "bbox": box,
                    "bbox_conf": box_conf,
                    "keypoints": person_kpts, # [17, 3] -> (x, y, conf)
                    "keypoint_format": "COCO17"
                })

        return {
            "backend": "yolo",
            "persons": tracked_persons,
            "latency_ms": latency_ms,
            "detected": len(tracked_persons) > 0,
            "connections": self.SKELETON_CONNECTIONS
        }


class MediaPipePoseTracker(BasePoseTracker):
    """
    Google MediaPipe 1.1 Tasks Pose Landmarker.
    Keypoints: 33 body landmarks with normalized 2D coordinates and metric 3D world landmarks.
    """
    SKELETON_CONNECTIONS = [
        # Torso & Arms
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        # Torso & Legs
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), # Left leg & foot
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32), # Right leg & foot
        # Head
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10)
    ]

    def __init__(self, model_path: str, min_conf: float = 0.5):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import mediapipe as mp

        logger.info(f"Initializing MediaPipe Pose Landmarker with model '{model_path}'...")
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, model_path)
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.mp_image = mp.Image
        logger.info("MediaPipe Pose Landmarker ready.")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        import mediapipe as mp
        t0 = time.perf_counter()
        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1].copy()
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.detector.detect(mp_img)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        h, w, _ = frame.shape
        tracked_persons = []
        if detection_result.pose_landmarks:
            for idx, (landmarks, world_landmarks) in enumerate(zip(detection_result.pose_landmarks, detection_result.pose_world_landmarks)):
                # Convert normalized landmarks to pixel coords
                kpts = []
                for lm in landmarks:
                    kpts.append([lm.x * w, lm.y * h, lm.visibility or 1.0])
                
                # World landmarks (in meters relative to body center)
                world_kpts = []
                for wlm in world_landmarks:
                    world_kpts.append([wlm.x, wlm.y, wlm.z, wlm.visibility or 1.0])

                tracked_persons.append({
                    "id": idx,
                    "bbox": None,
                    "keypoints": np.array(kpts, dtype=np.float32),
                    "world_keypoints": np.array(world_kpts, dtype=np.float32),
                    "keypoint_format": "MediaPipe33"
                })

        return {
            "backend": "mediapipe",
            "persons": tracked_persons,
            "latency_ms": latency_ms,
            "detected": len(tracked_persons) > 0,
            "connections": self.SKELETON_CONNECTIONS
        }


def create_tracker(config: Dict[str, Any]) -> BasePoseTracker:
    backend = config.get("backend", "yolo").lower()
    if backend == "yolo":
        yolo_cfg = config.get("yolo", {})
        device = config.get("device", "cuda:0")
        conf = config.get("confidence_threshold", 0.5)
        return YOLOPoseTracker(
            model_path=yolo_cfg.get("model_name", "models/yolo11n-pose.pt"),
            device=device,
            conf_thresh=conf,
            imgsz=yolo_cfg.get("imgsz", 640)
        )
    elif backend == "mediapipe":
        mp_cfg = config.get("mediapipe", {})
        conf = config.get("confidence_threshold", 0.5)
        return MediaPipePoseTracker(
            model_path=mp_cfg.get("model_path", "models/pose_landmarker_lite.task"),
            min_conf=conf
        )
    else:
        raise ValueError(f"Unsupported tracker backend: {backend}")
