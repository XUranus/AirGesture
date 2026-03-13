"""
下载 MediaPipe Gesture Recognizer 模型
"""

import urllib.request
import os

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
MODEL_PATH = "models/gesture_recognizer.task"


def download_model():
    """下载手势识别模型"""
    os.makedirs("models", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"模型已存在: {MODEL_PATH}")
        return

    print("正在下载 MediaPipe Gesture Recognizer 模型...")
    print(f"URL: {MODEL_URL}")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"模型下载完成: {MODEL_PATH}")
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n请手动下载模型:")
        print(f"1. 访问: {MODEL_URL}")
        print(f"2. 保存到: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    download_model()
