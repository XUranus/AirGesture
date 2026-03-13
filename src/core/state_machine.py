"""
手势状态机模块
"""

import time
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass


class State(Enum):
    """状态枚举"""
    IDLE = "IDLE"               # 待机状态
    RECOGNIZING = "RECOGNIZING" # 识别模式
    SENDING = "SENDING"         # 发送中
    RECEIVING = "RECEIVING"     # 接收中
    SWIPING = "SWIPING"         # 滑动处理中


@dataclass
class StateContext:
    """状态上下文，存储状态相关的临时数据"""
    recognizing_start_time: float = 0.0
    receiving_start_time: float = 0.0
    last_gesture_time: float = 0.0
    last_gesture: Optional[str] = None  # 上一个确认的手势（用于检测状态变化）


class GestureStateMachine:
    """手势状态机"""

    # 手势到动作的映射
    GESTURE_ACTIONS = {
        "Closed_Fist": "send",       # 握拳 → 发送
        "Open_Palm": "receive",      # 张开手掌 → 接收
        "Swipe_Left": "swipe_left",  # 向左滑动（预留）
        "Swipe_Right": "swipe_right", # 向右滑动（预留）
    }

    def __init__(
        self,
        on_send: Optional[Callable] = None,
        on_receive: Optional[Callable] = None,
        on_swipe_left: Optional[Callable] = None,
        on_swipe_right: Optional[Callable] = None,
        recognizing_timeout: float = 3.0,
        receiving_timeout: float = 30.0
    ):
        """
        初始化状态机

        Args:
            on_send: 发送动作回调
            on_receive: 接收动作回调
            on_swipe_left: 左滑动作回调
            on_swipe_right: 右滑动作回调
            recognizing_timeout: 识别模式超时（秒）
            receiving_timeout: 接收模式超时（秒）
        """
        self.state = State.IDLE
        self.context = StateContext()
        self.recognizing_timeout = recognizing_timeout
        self.receiving_timeout = receiving_timeout

        # 回调函数
        self.on_send = on_send
        self.on_receive = on_receive
        self.on_swipe_left = on_swipe_left
        self.on_swipe_right = on_swipe_right

        # 状态变化回调
        self._state_change_callback: Optional[Callable[[State, State], None]] = None

    def set_state_change_callback(self, callback: Callable[[State, State], None]):
        """设置状态变化回调"""
        self._state_change_callback = callback

    def process_hand_entered(self) -> str:
        """
        处理手进入画面事件

        Returns:
            动作名称
        """
        if self.state == State.IDLE:
            self._transition(State.RECOGNIZING)
            return "enter_recognizing"
        return "none"

    def process_hand_left(self) -> str:
        """
        处理手离开画面事件

        Returns:
            动作名称
        """
        if self.state == State.RECOGNIZING:
            # 手离开后返回 IDLE 状态，重置手势追踪
            self.context.last_gesture = None
            self._transition(State.IDLE)
            return "hand_left"
        return "none"

    def process_gesture(self, gesture: Optional[str]) -> Optional[str]:
        """
        处理手势状态变化

        Args:
            gesture: 当前识别的手势名称

        Returns:
            触发的动作名称，或 None
        """
        now = time.time()
        self.context.last_gesture_time = now

        # 只在 RECOGNIZING 状态下处理手势变化
        if self.state != State.RECOGNIZING:
            return None

        last = self.context.last_gesture

        # 检测状态变化
        if last == "Open_Palm" and gesture == "Closed_Fist":
            # 张开 → 握拳：触发发送
            self.context.last_gesture = gesture
            if self.on_send:
                self.on_send()
            return "send"

        elif last == "Closed_Fist" and gesture == "Open_Palm":
            # 握拳 → 张开：触发接收
            self.context.last_gesture = gesture
            if self.on_receive:
                self.on_receive()
            return "receive"

        # 更新上一个手势（只追踪有效手势）
        if gesture in ["Open_Palm", "Closed_Fist"]:
            self.context.last_gesture = gesture

        return None

    def process_timeout(self) -> Optional[str]:
        """
        处理超时检查

        Returns:
            动作名称或 None
        """
        now = time.time()

        if self.state == State.RECOGNIZING:
            # 检查识别模式超时（无手势）
            if now - self.context.recognizing_start_time > self.recognizing_timeout:
                self._transition(State.IDLE)
                return "timeout"

        elif self.state == State.RECEIVING:
            # 检查接收超时
            if now - self.context.receiving_start_time > self.receiving_timeout:
                self._transition(State.IDLE)
                return "receive_timeout"

        return None

    def _execute_action(self, action: str) -> str:
        """执行动作"""
        if action == "send":
            if self.on_send:
                self.on_send()
            # 保持在 RECOGNIZING 状态，继续检测下一次变化
            return "send"

        elif action == "receive":
            if self.on_receive:
                self.on_receive()
            # 保持在 RECOGNIZING 状态，继续检测下一次变化
            return "receive"

        elif action == "swipe_left" and self.on_swipe_left:
            self._transition(State.SWIPING)
            self.on_swipe_left()
            self._transition(State.RECOGNIZING)
            return "swiped_left"

        elif action == "swipe_right" and self.on_swipe_right:
            self._transition(State.SWIPING)
            self.on_swipe_right()
            self._transition(State.RECOGNIZING)
            return "swiped_right"

        return action

    def _transition(self, new_state: State):
        """状态转换"""
        old_state = self.state
        self.state = new_state

        # 进入状态时的初始化
        if new_state == State.RECOGNIZING:
            self.context.recognizing_start_time = time.time()
            self.context.last_gesture_time = time.time()
            self.context.last_gesture = None  # 重置手势追踪

        # 触发状态变化回调
        if self._state_change_callback:
            self._state_change_callback(old_state, new_state)

    def reset(self):
        """重置状态机"""
        self.state = State.IDLE
        self.context = StateContext()
