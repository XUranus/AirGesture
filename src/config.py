"""
AirGesture 配置文件
"""

import os
from pathlib import Path


class Config:
    """应用配置"""

    # 应用信息
    APP_NAME = "AirGesture"
    APP_VERSION = "1.0.0"

    # 手势识别
    GESTURE_MODEL_PATH = "models/gesture_recognizer.task"
    DEBOUNCE_THRESHOLD = 3          # 连续帧数确认（降低以提高响应速度）
    GESTURE_COOLDOWN = 1.0          # 动作冷却时间（秒）
    RECOGNIZING_TIMEOUT = 5.0       # 识别模式超时（秒）

    # 网络
    SERVICE_PORT = 9527
    SERVICE_TYPE = "_airgesture._tcp.local."
    TRANSFER_TIMEOUT = 30.0         # 传输超时（秒）

    # 文件
    SAVE_DIR = str(Path.home() / "AirGesture" / "received")
    TEMP_DIR = str(Path.home() / "AirGesture" / "temp")

    # UI
    SHOW_OVERLAY = True             # 显示状态叠加层
    OVERLAY_OPACITY = 0.7
    THUMBNAIL_SIZE = (300, 200)     # 缩略图大小
    THUMBNAIL_POSITION = "bottom-right"

    # 调试模式开关
    DEBUG_MODE = False

    # 调试模块配置
    DEBUG_SHOW_LANDMARKS = True
    DEBUG_SHOW_INFO_PANEL = True
    DEBUG_LOG_GESTURES = True
    DEBUG_TEST_MODE_DEFAULT = False

    @classmethod
    def ensure_dirs(cls):
        """确保必要的目录存在"""
        Path(cls.SAVE_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.TEMP_DIR).mkdir(parents=True, exist_ok=True)
