"""
手势识别模块 - MediaPipe Gesture Recognizer 封装
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GestureResult:
    """手势识别结果"""
    gesture: Optional[str]      # 手势名称，None 表示未识别
    confidence: float           # 置信度
    hand_detected: bool         # 是否检测到手
    handedness: Optional[str]   # 左右手 ("Left" 或 "Right")
    landmarks: Optional[List]   # 手部关键点


class GestureRecognizer:
    """MediaPipe Gesture Recognizer 封装"""

    # 手势名称映射
    GESTURE_NAMES = {
        "Closed_Fist": "握拳",
        "Open_Palm": "张开手掌",
        "Pointing_Up": "指向上方",
        "Thumb_Down": "拇指向下",
        "Thumb_Up": "拇指向上",
        "Victory": "V字手势",
        "ILoveYou": "我爱你",
    }

    def __init__(self, model_path: str):
        """
        初始化手势识别器

        Args:
            model_path: 模型文件路径
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1  # 只跟踪一只手
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def recognize(self, frame: np.ndarray) -> GestureResult:
        """
        识别帧中的手势

        Args:
            frame: BGR 格式的图像帧

        Returns:
            GestureResult: 识别结果
        """
        # 转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 识别
        result = self.recognizer.recognize(mp_image)

        # 解析结果
        if not result.hand_landmarks:
            return GestureResult(
                gesture=None,
                confidence=0.0,
                hand_detected=False,
                handedness=None,
                landmarks=None
            )

        # 获取手部关键点
        landmarks = result.hand_landmarks[0]

        # 获取左右手信息
        handedness = None
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name

        # 获取手势
        if result.gestures and result.gestures[0]:
            top_gesture = result.gestures[0][0]
            return GestureResult(
                gesture=top_gesture.category_name,
                confidence=top_gesture.score,
                hand_detected=True,
                handedness=handedness,
                landmarks=landmarks
            )

        # 检测到手但手势未知
        return GestureResult(
            gesture=None,
            confidence=0.0,
            hand_detected=True,
            handedness=handedness,
            landmarks=landmarks
        )

    def get_landmarks_pixel(self, frame: np.ndarray, landmarks: List) -> List[tuple]:
        """
        获取像素坐标的手部关键点

        Args:
            frame: 图像帧
            landmarks: 归一化的关键点列表

        Returns:
            像素坐标的关键点列表 [(x, y), ...]
        """
        h, w = frame.shape[:2]
        return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
