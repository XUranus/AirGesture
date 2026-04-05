"""
Hand landmark detection using MediaPipe.
"""

import os

import cv2
import numpy as np

from .log import log_info, log_warn, log_err


# Model download URL
HAND_LANDMARKER_MODEL = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model_file(path: str = HAND_LANDMARKER_MODEL) -> str:
    """
    Ensure the hand landmark model file exists, download if needed.

    Args:
        path: Path to save the model file

    Returns:
        Path to the model file
    """
    if os.path.exists(path):
        return path

    log_info(f"Downloading {path} ...")
    import requests

    r = requests.get(MODEL_URL, stream=True, timeout=120)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    return path


class HandDetector:
    """
    Hand landmark detector using MediaPipe.

    Supports two MediaPipe APIs:
    - tasks API (preferred, newer)
    - solutions API (fallback, older)
    """

    def __init__(self, static_mode: bool = True, max_hands: int = 1, min_conf: float = 0.5):
        """
        Initialize the hand detector.

        Args:
            static_mode: Whether to treat frames as static images
            max_hands: Maximum number of hands to detect
            min_conf: Minimum confidence threshold
        """
        self.api = None

        # Try tasks API first (newer)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision as mpv

            p = ensure_model_file()
            opts = mpv.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=p),
                num_hands=max_hands,
                min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=min_conf,
                min_tracking_confidence=0.5,
            )
            self.detector = mpv.HandLandmarker.create_from_options(opts)
            self.mp = mp
            self.api = "tasks"
            return
        except Exception as e:
            log_warn(f"MediaPipe tasks API failed: {e}")

        # Fallback to solutions API (older)
        try:
            import mediapipe as mp

            self.detector = mp.solutions.hands.Hands(
                static_image_mode=static_mode,
                max_num_hands=max_hands,
                min_detection_confidence=min_conf,
                min_tracking_confidence=0.5,
            )
            self.mp = mp
            self.api = "solutions"
            return
        except Exception as e:
            log_err(f"MediaPipe init failed: {e}")
            raise RuntimeError(f"MediaPipe init failed: {e}")

    def detect(self, frame_bgr):
        """
        Detect hand landmarks in a frame.

        Args:
            frame_bgr: Input frame in BGR format

        Returns:
            Numpy array of shape (63,) with flattened landmarks [x, y, z, x, y, z, ...],
            or None if no hand detected
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.api == "tasks":
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.detector.detect(img)
            if res.hand_landmarks and len(res.hand_landmarks) > 0:
                c = []
                for lm in res.hand_landmarks[0]:
                    c.extend([lm.x, lm.y, lm.z])
                return np.array(c, dtype=np.float32)
        else:
            res = self.detector.process(rgb)
            if res.multi_hand_landmarks:
                c = []
                for lm in res.multi_hand_landmarks[0].landmark:
                    c.extend([lm.x, lm.y, lm.z])
                return np.array(c, dtype=np.float32)

        return None

    def close(self):
        """Release resources."""
        if hasattr(self.detector, "close"):
            self.detector.close()
