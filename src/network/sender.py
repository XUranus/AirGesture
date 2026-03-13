"""
文件发送模块
"""

import socket
import json
import os
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class TransferProgress:
    """传输进度"""
    total_bytes: int
    transferred_bytes: int
    percentage: float


class FileSender:
    """文件发送器"""

    CHUNK_SIZE = 8192  # 8KB per chunk

    def __init__(self, host: str, port: int):
        """
        初始化发送器

        Args:
            host: 目标主机地址
            port: 目标端口
        """
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None

    def connect(self, timeout: float = 10.0) -> bool:
        """
        建立连接

        Args:
            timeout: 连接超时时间

        Returns:
            是否成功连接
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((self.host, self.port))

            # 等待确认
            ack = self.socket.recv(1024).decode()
            if not ack.startswith("ACK"):
                print(f"连接失败: 无效的响应 {ack}")
                return False

            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def send_file(
        self,
        filepath: str,
        progress_callback: Optional[Callable[[TransferProgress], None]] = None
    ) -> bool:
        """
        发送文件

        Args:
            filepath: 文件路径
            progress_callback: 进度回调函数

        Returns:
            是否成功发送
        """
        if not self.socket:
            print("未连接")
            return False

        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath)

        try:
            # 发送文件头
            header = json.dumps({
                "type": "FILE_HEADER",
                "filename": filename,
                "size": filesize,
                "timestamp": os.path.getmtime(filepath)
            })
            self.socket.send(header.encode())

            # 等待接收方准备就绪
            ready = self.socket.recv(1024).decode()
            if ready != "READY":
                print(f"接收方未就绪: {ready}")
                return False

            # 发送文件数据
            transferred = 0
            with open(filepath, 'rb') as f:
                while transferred < filesize:
                    chunk = f.read(self.CHUNK_SIZE)
                    if not chunk:
                        break
                    self.socket.sendall(chunk)
                    transferred += len(chunk)

                    if progress_callback:
                        progress = TransferProgress(
                            total_bytes=filesize,
                            transferred_bytes=transferred,
                            percentage=transferred / filesize * 100
                        )
                        progress_callback(progress)

            # 等待确认
            ack = self.socket.recv(1024).decode()
            if not ack.startswith("ACK"):
                print(f"发送失败: {ack}")
                return False

            # 发送完成信号
            self.socket.send(b"DONE")
            confirm = self.socket.recv(1024).decode()

            return confirm == "CONFIRM"

        except Exception as e:
            print(f"发送文件失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.socket:
            try:
                self.socket.send(b"CLOSE")
            except:
                pass
            finally:
                self.socket.close()
                self.socket = None
