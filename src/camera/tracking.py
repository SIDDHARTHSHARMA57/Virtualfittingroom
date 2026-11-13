"""
Tracking Module.
Performs pose, hand, and face landmark inference with independent timing,
temporal smoothing, confidence thresholds, and missing landmark handling.
"""
import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)

class OneEuroFilter:
    """
    1-Euro filter for smooth low-jitter human motion tracking with low lag.
    """
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    def _smoothing_factor(self, te: float, cutoff: np.ndarray) -> np.ndarray:
        r = 2 * np.pi * cutoff * te
        return r / (r + 1)

    def filter(self, x: np.ndarray, timestamp: float) -> np.ndarray:
        if self.t_prev is None or self.x_prev is None:
            self.x_prev = np.array(x, dtype=np.float32)
            self.dx_prev = np.zeros_like(x, dtype=np.float32)
            self.t_prev = timestamp
            return self.x_prev

        te = max(timestamp - self.t_prev, 1e-5)
        self.t_prev = timestamp

        # Filtered derivative
        a_d = self._smoothing_factor(te, np.full_like(x, self.d_cutoff))
        dx = (x - self.x_prev) / te
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Filtered signal
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(te, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class UnifiedTracker:
    """
    Unified multi-model tracker managing MediaPipe Pose, Hands, and Face Landmarkers.
    Supports toggling hands/face on and off, separate latency measurement, and confidence filtering.
    """
    def __init__(
        self,
        pose_model_path: str,
        hand_model_path: str,
        face_model_path: str,
        enable_pose: bool = True,
        enable_hands: bool = True,
        enable_face: bool = True,
        min_pose_conf: float = 0.5,
        min_hand_conf: float = 0.45,
        min_face_conf: float = 0.45,
        enable_smoothing: bool = True,
    ):
        self.enable_pose = enable_pose
        self.enable_hands = enable_hands
        self.enable_face = enable_face
        self.enable_smoothing = enable_smoothing

        self.pose_filter = OneEuroFilter() if enable_smoothing else None
        # World landmarks are in metres (not 0..1), so velocity-adaptive beta is larger
        self.world_filter = OneEuroFilter(min_cutoff=1.5, beta=0.5) if enable_smoothing else None
        self.hand_filters: Dict[str, OneEuroFilter] = {}
        self._last_ts_ms = -1
        # person lock (image-space torso centre / shoulder width of the tracked customer)
        self._track_center: Optional[np.ndarray] = None
        self._track_width = 0.2
        self._lost_frames = 0

        # Verify model files
        for name, path in [("Pose", pose_model_path), ("Hand", hand_model_path), ("Face", face_model_path)]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing {name} model at {path}")

        # Pose Landmarker
        self.pose_detector = None
        if enable_pose:
            pose_opts = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=2,  # >1 so we can lock onto the customer, not whoever MediaPipe ranks first
                min_pose_detection_confidence=min_pose_conf,
                min_pose_presence_confidence=min_pose_conf,
                min_tracking_confidence=min_pose_conf,
                output_segmentation_masks=False,
            )
            self.pose_detector = vision.PoseLandmarker.create_from_options(pose_opts)

        # Hand Landmarker
        self.hand_detector = None
        if enable_hands:
            hand_opts = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=4,  # extra slots so a bystander's hand can't crowd out the customer's
                min_hand_detection_confidence=min_hand_conf,
                min_hand_presence_confidence=min_hand_conf,
                min_tracking_confidence=min_hand_conf,
            )
            self.hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

        # Face Landmarker
        self.face_detector = None
        if enable_face:
            face_opts = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=face_model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=2,
                min_face_detection_confidence=min_face_conf,
                min_face_presence_confidence=min_face_conf,
                min_tracking_confidence=min_face_conf,
                output_facial_transformation_matrixes=True,
                output_face_blendshapes=True,
            )
            self.face_detector = vision.FaceLandmarker.create_from_options(face_opts)

        # Telemetry
        self.last_pose_latency_ms: float = 0.0
        self.last_hand_latency_ms: float = 0.0
        self.last_face_latency_ms: float = 0.0

    def process(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Runs tracking pipelines on the given BGR image frame.
        Returns extracted landmark dictionaries and latency timings.
        """
        ts = timestamp if timestamp is not None else time.time()
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode tracks between frames (skips re-detection); needs strictly increasing ms timestamps
        ts_ms = max(int(ts * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        body_points: List[List[float]] = []
        world_points: List[List[float]] = []
        hand_data: Dict[str, List[List[float]]] = {}
        hand_world: Dict[str, List[List[float]]] = {}
        face_points: List[List[float]] = []
        face_matrix: Optional[List[List[float]]] = None
        blendshapes: Dict[str, float] = {}

        # 1. Pose Inference
        if self.enable_pose and self.pose_detector is not None:
            t0 = time.perf_counter()
            pose_res = self.pose_detector.detect_for_video(mp_image, ts_ms)
            self.last_pose_latency_ms = (time.perf_counter() - t0) * 1000.0

            sel = self._select_person(pose_res.pose_landmarks)
            if sel is not None:
                raw_kpts = []
                for lm in pose_res.pose_landmarks[sel]:
                    conf = lm.visibility if lm.visibility is not None else 1.0
                    raw_kpts.append([float(lm.x), float(lm.y), float(conf)])
                raw_arr = np.array(raw_kpts, dtype=np.float32)

                if self.enable_smoothing and self.pose_filter is not None:
                    # Smooth x and y, preserve confidence
                    xy = raw_arr[:, :2]
                    smoothed_xy = self.pose_filter.filter(xy, ts)
                    body_points = [
                        [float(smoothed_xy[i, 0]), float(smoothed_xy[i, 1]), float(raw_arr[i, 2])]
                        for i in range(len(raw_arr))
                    ]
                else:
                    body_points = [[float(p[0]), float(p[1]), float(p[2])] for p in raw_arr]

            if sel is not None and pose_res.pose_world_landmarks:
                w = np.array([[l.x, l.y, l.z, l.visibility if l.visibility is not None else 1.0]
                              for l in pose_res.pose_world_landmarks[sel]], dtype=np.float32)
                if self.world_filter is not None:
                    w[:, :3] = self.world_filter.filter(w[:, :3], ts)
                world_points = w.tolist()
            elif self.world_filter is not None:
                self.world_filter.reset()  # person lost: don't smear old pose into the next detection

        # 2. Hand Inference
        if self.enable_hands and self.hand_detector is not None:
            t0 = time.perf_counter()
            hand_res = self.hand_detector.detect_for_video(mp_image, ts_ms)
            self.last_hand_latency_ms = (time.perf_counter() - t0) * 1000.0

            for idx, landmarks in enumerate(hand_res.hand_landmarks or []):
                if not self._hand_belongs(landmarks, body_points):
                    continue  # someone else's hand
                category = "Unknown"
                if idx < len(hand_res.handedness):
                    cats = hand_res.handedness[idx]
                    if cats:
                        category = cats[0].category_name
                if category in hand_data:  # both hands got the same label
                    category = category + "2"

                hand_data[category] = [
                    [float(p.x), float(p.y), float(p.z)]
                    for p in landmarks
                ]
                # Metric 3D hand landmarks (camera-aligned axes, hand-centred) drive SMPL-X fingers
                if hand_res.hand_world_landmarks and idx < len(hand_res.hand_world_landmarks):
                    hw = np.array([[p.x, p.y, p.z] for p in hand_res.hand_world_landmarks[idx]], dtype=np.float32)
                    if self.enable_smoothing:
                        f = self.hand_filters.setdefault(category, OneEuroFilter(min_cutoff=1.5, beta=0.5))
                        hw = f.filter(hw, ts)
                    hand_world[category] = hw.tolist()
            for cat in list(self.hand_filters):
                if cat not in hand_world:
                    del self.hand_filters[cat]

        # 3. Face Inference
        if self.enable_face and self.face_detector is not None:
            t0 = time.perf_counter()
            face_res = self.face_detector.detect_for_video(mp_image, ts_ms)
            self.last_face_latency_ms = (time.perf_counter() - t0) * 1000.0

            fi = self._select_face(face_res.face_landmarks, body_points)
            if fi is not None:
                face_points = [
                    [float(p.x), float(p.y), float(p.z)]
                    for p in face_res.face_landmarks[fi]
                ]

            if fi is not None and face_res.facial_transformation_matrixes:
                face_matrix = face_res.facial_transformation_matrixes[fi].tolist()

            if fi is not None and face_res.face_blendshapes:
                for item in face_res.face_blendshapes[fi]:
                    blendshapes[item.category_name] = float(item.score)

        return {
            "timestamp": ts,
            "body_points": body_points,
            "world_points": world_points,
            "hands": hand_data,
            "hands_world": hand_world,
            "face": face_points,
            "face_matrix": face_matrix,
            "blendshapes": blendshapes,
            "latencies": {
                "pose_ms": self.last_pose_latency_ms,
                "hand_ms": self.last_hand_latency_ms,
                "face_ms": self.last_face_latency_ms,
                "total_ms": self.last_pose_latency_ms + self.last_hand_latency_ms + self.last_face_latency_ms
            }
        }

    # ---------------- person lock ----------------
    @staticmethod
    def _torso(lms) -> Tuple[np.ndarray, float]:
        """Image-space torso centre and shoulder width of one pose."""
        p = np.array([[lms[i].x, lms[i].y] for i in (11, 12, 23, 24)], dtype=np.float32)
        return p.mean(axis=0), float(np.linalg.norm(p[0] - p[1]))

    def _select_person(self, poses) -> Optional[int]:
        """Lock onto one person. First lock: the largest torso (nearest the mirror).
        Afterwards: the pose closest to where the locked person was last frame."""
        if not poses:
            self._lost_frames = self._lost_frames + 1
            if self._lost_frames > 15:  # ~2 s gone: release the lock
                self._track_center = None
            return None
        self._lost_frames = 0
        torsos = [self._torso(p) for p in poses]
        prev = self._track_center
        if prev is None:
            idx = int(np.argmax([w for _, w in torsos]))
        else:
            idx = int(np.argmin([np.linalg.norm(c - prev) for c, _ in torsos]))
            if np.linalg.norm(torsos[idx][0] - prev) > 0.25:  # big jump = different person; reset filters
                for f in (self.pose_filter, self.world_filter):
                    if f is not None:
                        f.reset()
        self._track_center, self._track_width = torsos[idx]
        return idx

    def _hand_belongs(self, hand_lms, body_points) -> bool:
        """Keep a hand only if its wrist is near one of the locked person's pose wrists."""
        if not body_points:
            return True  # no body to compare against (hands-only use)
        wrist = np.array([hand_lms[0].x, hand_lms[0].y])
        hand_len = np.linalg.norm(wrist - np.array([hand_lms[9].x, hand_lms[9].y]))
        # ponytail: fixed image-space gate; tune 'reach' if fast hands get dropped
        reach = max(0.12, 0.6 * self._track_width, 1.5 * hand_len)
        d = min(np.linalg.norm(wrist - np.array(body_points[i][:2])) for i in (15, 16))
        return d < reach

    def _select_face(self, faces, body_points) -> Optional[int]:
        """Face nearest the locked person's nose; none if no face is plausibly theirs."""
        if not faces:
            return None
        if not body_points:
            return 0
        nose = np.array(body_points[0][:2])
        d = [np.linalg.norm(np.array([f[1].x, f[1].y]) - nose) for f in faces]
        i = int(np.argmin(d))
        return i if d[i] < max(0.15, self._track_width) else None

    def close(self):
        for detector in (self.pose_detector, self.hand_detector, self.face_detector):
            if detector is not None:
                try:
                    detector.close()
                except Exception as e:
                    logger.debug(f"Error closing detector: {e}")
        logger.info("Unified tracker closed.")
