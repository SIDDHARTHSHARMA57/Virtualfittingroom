import cv2
import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class CameraManager:
    """
    Manages robust camera capture using OpenCV with reconnect logic and error handling.
    """
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480, target_fps: int = 30, buffer_size: int = 1):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        """Initializes the video capture device."""
        logger.info(f"Connecting to camera index {self.device_index}...")
        self.cap = cv2.VideoCapture(self.device_index)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video capture device at index {self.device_index}.")
            return False

        # Set hardware properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened successfully: {int(actual_w)}x{int(actual_h)} @ {actual_fps:.1f} FPS")
        return True

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Reads a frame safely. Returns (success, frame)."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Empty frame received or camera stream interrupted.")
            return False, None
        return True, frame

    def release(self):
        """Releases the camera resource."""
        if self.cap is not None:
            logger.info("Releasing camera...")
            self.cap.release()
            self.cap = None
