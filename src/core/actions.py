"""
动作执行模块 - 截图、发送、接收等
"""

import os
import time
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

from src.config import Config


def take_screenshot(save_dir: Optional[str] = None) -> str:
    """
    截取当前屏幕

    Args:
        save_dir: 保存目录，默认使用配置中的临时目录

    Returns:
        截图文件路径
    """
    import mss

    save_dir = save_dir or Config.TEMP_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)

    with mss.mss() as sct:
        # 截取所有显示器
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        screenshot = sct.grab(monitor)

        # 保存
        from PIL import Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        img.save(filepath)

    return filepath


def show_notification(title: str, message: str):
    """
    显示系统通知

    Args:
        title: 通知标题
        message: 通知内容
    """
    try:
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(title, message, duration=3)
    except ImportError:
        # 如果没有安装 win10toast，使用简单的控制台输出
        print(f"[通知] {title}: {message}")
    except Exception as e:
        print(f"显示通知失败: {e}")


class SendAction:
    """发送动作处理"""

    def __init__(self):
        self.thumbnail_window = None
        self._screenshot_path: Optional[str] = None

    def execute(self, devices: list, on_progress: Optional[Callable] = None) -> bool:
        """
        执行发送动作

        Args:
            devices: 可用设备列表
            on_progress: 进度回调函数

        Returns:
            是否成功
        """
        # 1. 截图
        self._screenshot_path = take_screenshot()

        # 2. 检查设备
        if not devices:
            show_notification("AirGesture", "未发现可用设备")
            return False

        # 3. 如果只有一个设备，自动选择
        target = devices[0]

        # 4. 发送文件
        from src.network.sender import FileSender

        sender = FileSender(target['ip'], target['port'])

        if sender.connect():
            success = sender.send_file(self._screenshot_path, on_progress)
            sender.close()

            if success:
                show_notification("AirGesture", f"发送成功 → {target['name']}")
            else:
                show_notification("AirGesture", "发送失败")
            return success
        else:
            show_notification("AirGesture", f"无法连接到 {target['name']}")
            return False

    def get_screenshot_path(self) -> Optional[str]:
        """获取截图路径"""
        return self._screenshot_path


class ReceiveAction:
    """接收动作处理"""

    def __init__(self):
        self._received_path: Optional[str] = None

    def execute(self, on_received: Optional[Callable] = None):
        """
        执行接收动作

        Args:
            on_received: 接收完成回调
        """
        show_notification("AirGesture", "等待接收文件...")

        from src.network.receiver import FileReceiver
        from src.ui.image_viewer import ImageViewer

        receiver = FileReceiver(Config.SERVICE_PORT, Config.SAVE_DIR)
        receiver.start()

        filepath = receiver.wait_for_file()

        if filepath:
            self._received_path = filepath
            show_notification("AirGesture", f"已接收: {Path(filepath).name}")

            # 自动打开图片
            ImageViewer.open_image(filepath)

            if on_received:
                on_received(filepath)

    def get_received_path(self) -> Optional[str]:
        """获取接收的文件路径"""
        return self._received_path
