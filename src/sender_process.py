"""
发送进程 - 独立进程处理文件发送
"""

import sys
import socket
import json
import os
import time


def run_sender(filepath: str, port: int):
    """运行发送服务器"""
    # 先创建 socket 服务器（确保端口已绑定）
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)
    server.settimeout(120.0)  # 120秒超时

    print(f"[发送进程] 等待连接，端口 {port}...")

    # 显示缩略图
    from src.ui.thumbnail_window import ThumbnailWindow
    thumbnail = ThumbnailWindow(position="bottom-right", size=(300, 200))
    thumbnail.show(filepath)

    time.sleep(0.3)  # 等待窗口初始化
    thumbnail.update_progress(0, "等待连接...")

    client = None
    try:
        client, addr = server.accept()
        print(f"[发送进程] 连接来自: {addr}")
        thumbnail.update_progress(10, "已连接，准备发送...")

        # 发送确认
        client.send(b"ACK:SENDER")

        # 等待接收方就绪
        ready = client.recv(1024).decode()
        if ready != "READY":
            print(f"[发送进程] 接收方未就绪")
            thumbnail.show_error("接收方未就绪")
            return

        # 发送文件头
        filesize = os.path.getsize(filepath)
        filename = os.path.basename(filepath)
        header = json.dumps({
            "type": "FILE_HEADER",
            "filename": filename,
            "size": filesize
        })
        client.send(header.encode())

        # 等待确认
        ack = client.recv(1024).decode()
        if not ack.startswith("ACK"):
            print(f"[发送进程] 接收方确认失败: {ack}")
            thumbnail.show_error("确认失败")
            return

        # 发送文件数据
        transferred = 0
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                client.sendall(chunk)
                transferred += len(chunk)
                progress = 10 + (transferred / filesize) * 80
                thumbnail.update_progress(progress, f"发送中 {progress:.0f}%")

        # 等待完成确认
        done = client.recv(1024).decode()
        if done == "DONE":
            client.send(b"CONFIRM")
            print(f"[发送进程] 发送完成: {filename}")
            thumbnail.show_success()

            # 等待一段时间让用户看到成功状态，并确保消息被处理
            for _ in range(40):  # 4秒，每100ms检查一次
                if thumbnail.is_closed():
                    break
                time.sleep(0.1)
            time.sleep(1.0)  # 额外等待淡出动画
        else:
            thumbnail.show_error("传输未完成")
            time.sleep(2.0)

    except socket.timeout:
        print("[发送进程] 等待连接超时")
        thumbnail.show_error("等待连接超时")
        time.sleep(2.0)
    except Exception as e:
        print(f"[发送进程] 错误: {e}")
        thumbnail.show_error(str(e))
        time.sleep(2.0)
    finally:
        if client:
            client.close()
        server.close()
        # 确保窗口关闭
        if not thumbnail.is_closed():
            thumbnail._close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python -m src.sender_process <filepath> <port>")
        sys.exit(1)

    run_sender(sys.argv[1], int(sys.argv[2]))
