"""
摄像头手势调试模块
"""

import cv2
import numpy as np
import time
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass
import json


@dataclass
class DebugFrame:
    """调试帧数据"""
    timestamp: str
    frame: np.ndarray
    hand_detected: bool
    gesture: Optional[str]
    confidence: float
    state: str
    fps: float


class GestureDebugger:
    """手势识别调试器"""

    # 手部关键点连接（用于绘制骨架）
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),      # 食指
        (0, 9), (9, 10), (10, 11), (11, 12), # 中指
        (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20), # 小指
        (5, 9), (9, 13), (13, 17)            # 手掌
    ]

    # 手势名称中文映射
    GESTURE_NAMES = {
        "Closed_Fist": "握拳",
        "Open_Palm": "张开手掌",
        "Pointing_Up": "指向上方",
        "Thumb_Down": "拇指向下",
        "Thumb_Up": "拇指向上",
        "Victory": "V字手势",
        "ILoveYou": "我爱你",
        None: "未识别"
    }

    # 状态颜色
    STATE_COLORS = {
        'IDLE': (128, 128, 128),       # 灰色
        'RECOGNIZING': (0, 255, 0),    # 绿色
        'SENDING': (0, 165, 255),      # 橙色
        'RECEIVING': (255, 0, 0),      # 蓝色
        'SWIPING': (0, 255, 255)       # 黄色
    }

    def __init__(self):
        self.running = False
        self.test_mode = False  # 测试模式，不触发实际动作
        self.logs: List[dict] = []
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self._last_hand_detected = False

    def start(self):
        """启动调试模式"""
        self.running = True
        cv2.namedWindow('AirGesture Debug', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('AirGesture Debug', 1200, 800)

    def stop(self):
        """停止调试模式"""
        self.running = False
        cv2.destroyWindow('AirGesture Debug')

    def process_frame(
        self,
        frame: np.ndarray,
        hand_detected: bool,
        gesture: Optional[str],
        confidence: float,
        state: str,
        landmarks: Optional[List] = None
    ) -> np.ndarray:
        """
        处理帧并绘制调试信息

        Args:
            frame: 原始帧
            hand_detected: 是否检测到手
            gesture: 识别的手势
            confidence: 置信度
            state: 当前状态
            landmarks: 手部关键点列表（可选）

        Returns:
            带有调试信息的帧
        """
        self.frame_count += 1

        # 计算FPS
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now

        # 创建调试帧副本
        debug_frame = frame.copy()
        h, w = debug_frame.shape[:2]

        # 绘制手部关键点和骨架
        if landmarks:
            self._draw_hand_landmarks(debug_frame, landmarks, w, h)

        # 绘制状态信息面板
        self._draw_info_panel(debug_frame, hand_detected, gesture, confidence, state)

        # 绘制状态指示器
        self._draw_state_indicator(debug_frame, state)

        # 记录日志
        self._log_frame(hand_detected, gesture, confidence, state)

        return debug_frame

    def _draw_hand_landmarks(self, frame: np.ndarray, landmarks: List, w: int, h: int):
        """绘制手部关键点和骨架"""
        # 将归一化坐标转换为像素坐标
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # 绘制骨架连接线
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            start = points[start_idx]
            end = points[end_idx]
            cv2.line(frame, start, end, (0, 255, 0), 2)

        # 绘制关键点
        for i, point in enumerate(points):
            # 不同颜色区分不同部位
            if i == 0:  # 手腕
                color = (0, 0, 255)
            elif i in [4, 8, 12, 16, 20]:  # 指尖
                color = (255, 0, 0)
            else:
                color = (0, 255, 255)

            cv2.circle(frame, point, 5, color, -1)
            cv2.putText(frame, str(i), (point[0] + 5, point[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_info_panel(self, frame: np.ndarray, hand_detected: bool,
                         gesture: Optional[str], confidence: float, state: str):
        """绘制信息面板"""
        h, w = frame.shape[:2]

        # 创建半透明面板
        panel = np.zeros((150, 250, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # 深灰色背景

        # 状态信息
        y_offset = 25
        texts = [
            f"State: {state}",
            f"Gesture: {self.GESTURE_NAMES.get(gesture, gesture or 'Unknown')}",
            f"Confidence: {confidence:.2f}" if gesture else "Confidence: --",
            f"Hand: {'Yes' if hand_detected else 'No'}",
            f"FPS: {self.fps:.1f}"
        ]

        for text in texts:
            cv2.putText(panel, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        # 将面板叠加到帧上
        overlay = frame.copy()
        overlay[10:160, w-260:w-10] = panel
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    def _draw_state_indicator(self, frame: np.ndarray, state: str):
        """绘制状态指示器（左上角）"""
        color = self.STATE_COLORS.get(state, (255, 255, 255))

        # 绘制状态圆点
        cv2.circle(frame, (30, 30), 15, color, -1)
        cv2.putText(frame, state, (50, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 测试模式指示
        if self.test_mode:
            cv2.putText(frame, "[TEST MODE]", (50, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def _log_frame(self, hand_detected: bool, gesture: Optional[str],
                   confidence: float, state: str):
        """记录帧日志"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # 只在有意义的事件时记录
        if gesture or hand_detected != self._last_hand_detected:
            log_entry = {
                "timestamp": timestamp,
                "hand_detected": hand_detected,
                "gesture": gesture,
                "confidence": confidence,
                "state": state
            }
            self.logs.append(log_entry)
            print(f"[{timestamp}] gesture={gesture}, conf={confidence:.2f}, state={state}")

        self._last_hand_detected = hand_detected

    def show_debug_window(self, frame: np.ndarray) -> bool:
        """
        显示调试窗口

        Returns:
            是否应该继续运行（False 表示用户关闭了窗口）
        """
        cv2.imshow('AirGesture Debug', frame)

        # 检查窗口是否被关闭
        if cv2.getWindowProperty('AirGesture Debug', cv2.WND_PROP_VISIBLE) < 1:
            self.running = False
            return False

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
            return False
        elif key == ord('t'):
            self.test_mode = not self.test_mode
            print(f"Test mode: {'ON' if self.test_mode else 'OFF'}")
        elif key == ord('s'):
            self._save_screenshot(frame)
        elif key == ord('e'):
            self._export_logs()

        return self.running

    def _save_screenshot(self, frame: np.ndarray):
        """保存截图"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

    def _export_logs(self):
        """导出日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_logs_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
        print(f"Logs exported: {filename}")

    def get_logs(self) -> List[dict]:
        """获取日志列表"""
        return self.logs

    def clear_logs(self):
        """清除日志"""
        self.logs = []

    def is_running(self) -> bool:
        """检查是否在运行"""
        return self.running
