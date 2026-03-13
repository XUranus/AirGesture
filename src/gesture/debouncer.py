"""
手势防抖模块
"""

import time
from typing import Optional


class GestureDebouncer:
    """手势防抖处理"""

    def __init__(self, threshold: int = 5, cooldown: float = 0.5):
        """
        初始化防抖器

        Args:
            threshold: 连续相同手势的帧数阈值
            cooldown: 动作冷却时间（秒）
        """
        self.threshold = threshold
        self.cooldown = cooldown
        self.last_gesture: Optional[str] = None
        self.gesture_count = 0
        self.last_action_time = 0.0

        # 用于检测"手存在"状态变化
        self._last_hand_detected = False

    def process(self, current_gesture: Optional[str]) -> Optional[str]:
        """
        处理当前手势，返回确认的手势

        Args:
            current_gesture: 当前识别的手势，None 表示未检测到手或未识别

        Returns:
            确认的手势名称，或 None
        """
        now = time.time()

        # 冷却期内不处理
        if now - self.last_action_time < self.cooldown:
            return None

        # 统计连续相同手势
        if current_gesture == self.last_gesture:
            self.gesture_count += 1
        else:
            self.last_gesture = current_gesture
            self.gesture_count = 1

        # 达到阈值确认手势
        if self.gesture_count >= self.threshold and current_gesture is not None:
            self.last_action_time = now
            self.gesture_count = 0
            return current_gesture

        return None

    def reset(self):
        """重置状态"""
        self.last_gesture = None
        self.gesture_count = 0

    def check_hand_state_change(self, hand_detected: bool) -> Optional[str]:
        """
        检查手的存在状态是否发生变化

        Args:
            hand_detected: 当前是否检测到手

        Returns:
            "hand_entered" 或 "hand_left" 或 None
        """
        if hand_detected and not self._last_hand_detected:
            self._last_hand_detected = True
            return "hand_entered"
        elif not hand_detected and self._last_hand_detected:
            self._last_hand_detected = False
            return "hand_left"

        return None
