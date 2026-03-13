"""
AirGesture 主程序入口
"""

import cv2
import argparse
import subprocess
import sys
from pathlib import Path

from src.config import Config
from src.gesture.recognizer import GestureRecognizer, GestureResult
from src.gesture.debouncer import GestureDebouncer
from src.core.state_machine import GestureStateMachine, State
from src.core.actions import take_screenshot, show_notification
from src.network.discovery import DeviceDiscovery


class AirGestureApp:
    """AirGesture 应用主类"""

    def __init__(self, debug_mode: bool = False):
        """
        初始化应用

        Args:
            debug_mode: 是否启用调试模式
        """
        self.debug_mode = debug_mode

        # 确保目录存在
        Config.ensure_dirs()

        # 初始化手势识别
        self.recognizer = None
        self.debouncer = GestureDebouncer(
            threshold=Config.DEBOUNCE_THRESHOLD,
            cooldown=Config.GESTURE_COOLDOWN
        )

        # 初始化状态机
        self.state_machine = GestureStateMachine(
            on_send=self._on_send,
            on_receive=self._on_receive,
            recognizing_timeout=Config.RECOGNIZING_TIMEOUT
        )

        # 初始化网络发现
        self.discovery = DeviceDiscovery(port=Config.SERVICE_PORT)

        # 发送进程端口
        self.sender_port = Config.SERVICE_PORT + 1

        # 调试模块
        self.debugger = None
        if debug_mode:
            from src.debug.gesture_debugger import GestureDebugger
            self.debugger = GestureDebugger()

        # 摄像头
        self.cap = None
        self.running = False

    def _init_recognizer(self):
        """初始化手势识别器"""
        model_path = Config.GESTURE_MODEL_PATH
        if not Path(model_path).exists():
            print(f"错误: 模型文件不存在: {model_path}")
            print("请运行 python download_model.py 下载模型")
            return False

        try:
            self.recognizer = GestureRecognizer(model_path)
            print(f"手势识别器已加载: {model_path}")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

    def start(self):
        """启动应用"""
        print("=" * 50)
        print("AirGesture 启动中...")
        print("=" * 50)

        # 初始化识别器
        if not self._init_recognizer():
            return

        # 启动网络发现
        self.discovery.start()

        # 启动摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            print("错误: 无法打开摄像头")
            return

        self.running = True

        if self.debug_mode:
            self.debugger.start()
            print("调试模式已启用")

        show_notification("AirGesture", "应用已启动")
        print("手势操作：张开→握拳=发送，握拳→张开=接收")

        # 主循环
        self._main_loop()

    def _main_loop(self):
        """主循环"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 识别手势
            result = self.recognizer.recognize(frame)

            # 检查手的存在状态变化
            hand_state = self.debouncer.check_hand_state_change(result.hand_detected)
            if hand_state == "hand_entered":
                self.state_machine.process_hand_entered()
            elif hand_state == "hand_left":
                self.state_machine.process_hand_left()

            # 处理超时
            self.state_machine.process_timeout()

            # 防抖处理手势
            confirmed_gesture = self.debouncer.process(result.gesture)

            # 状态机处理确认的手势
            if confirmed_gesture:
                self.state_machine.process_gesture(confirmed_gesture)

            # 调试模式
            if self.debug_mode:
                landmarks = result.landmarks if result.hand_detected else None
                debug_frame = self.debugger.process_frame(
                    frame,
                    result.hand_detected,
                    result.gesture,
                    result.confidence,
                    self.state_machine.state.value,
                    landmarks
                )
                if not self.debugger.show_debug_window(debug_frame):
                    break  # 用户关闭了窗口

                # 测试模式下不执行实际动作
                if self.debugger.test_mode:
                    continue

        self._cleanup()

    def _on_send(self):
        """发送动作回调 - 启动发送进程"""
        # 截图
        screenshot_path = take_screenshot()
        print(f"[发送] 截图已保存: {screenshot_path}")

        # 启动发送进程
        subprocess.Popen([
            sys.executable, "-m", "src.sender_process",
            screenshot_path, str(self.sender_port)
        ])

        print(f"[发送] 发送进程已启动，端口 {self.sender_port}")

    def _on_receive(self):
        """接收动作回调 - 启动接收进程"""
        # 获取发现的设备
        devices = self.discovery.get_devices()
        if not devices:
            show_notification("AirGesture", "未发现可用设备")
            print("[接收] 未发现可用设备")
            return

        target_ip = devices[0].ip

        # 启动接收进程
        subprocess.Popen([
            sys.executable, "-m", "src.receiver_process",
            target_ip, str(self.sender_port), Config.SAVE_DIR
        ])

        print(f"[接收] 接收进程已启动，连接 {target_ip}:{self.sender_port}")

    def _cleanup(self):
        """清理资源"""
        self.running = False

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        if self.debugger:
            self.debugger.stop()

        self.discovery.stop()
        print("AirGesture 已退出")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AirGesture - 手势隔空取物应用')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    args = parser.parse_args()

    app = AirGestureApp(debug_mode=args.debug)
    app.start()


if __name__ == "__main__":
    main()
