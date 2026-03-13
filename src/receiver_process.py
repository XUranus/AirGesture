"""
接收进程 - 独立进程处理文件接收
"""

import sys
import socket
import json
import os
import time


def run_receiver(host: str, port: int, save_dir: str):
    """连接并发送接收请求"""
    from src.config import Config
    from src.ui.image_viewer import ImageViewer

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 连接到发送方（带重试）
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(30.0)

    # 重试连接
    max_retries = 5
    for attempt in range(max_retries):
        try:
            client.connect((host, port))
            print(f"[接收进程] 已连接到 {host}:{port}")
            break
        except ConnectionRefusedError:
            if attempt < max_retries - 1:
                print(f"[接收进程] 连接被拒绝，等待重试 ({attempt + 1}/{max_retries})...")
                time.sleep(1.0)
            else:
                print("[接收进程] 连接失败，发送方未就绪")
                return

    try:
        # 等待确认
        ack = client.recv(1024).decode()
        if not ack.startswith("ACK"):
            print(f"[接收进程] 连接失败，收到: {ack}")
            return

        # 发送就绪信号
        client.send(b"READY")

        # 接收文件头
        header_data = client.recv(1024).decode()
        if not header_data:
            print("[接收进程] 未收到文件头")
            return
        try:
            header = json.loads(header_data)
        except json.JSONDecodeError as e:
            print(f"[接收进程] 文件头解析失败: {e}, 数据: {repr(header_data)}")
            return

        if header.get("type") != "FILE_HEADER":
            print("[接收进程] 无效的文件头")
            return

        filename = header["filename"]
        filesize = header["size"]
        print(f"[接收进程] 准备接收: {filename} ({filesize} bytes)")

        # 发送确认
        client.send(f"ACK:{filesize}".encode())

        # 接收文件
        filepath = os.path.join(save_dir, filename)
        received = 0

        with open(filepath, 'wb') as f:
            while received < filesize:
                chunk = client.recv(8192)
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)

        print(f"[接收进程] 接收完成: {filepath}")

        # 发送完成信号
        client.send(b"DONE")

        # 等待确认
        confirm = client.recv(1024).decode()
        if confirm == "CONFIRM":
            print(f"[接收进程] 打开图片: {filepath}")
            ImageViewer.open_image(filepath)

    except Exception as e:
        print(f"[接收进程] 错误: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python -m src.receiver_process <host> <port> <save_dir>")
        sys.exit(1)

    run_receiver(sys.argv[1], int(sys.argv[2]), sys.argv[3])
