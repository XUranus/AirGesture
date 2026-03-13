"""
文件接收模块
"""

import socket
import json
import os
from pathlib import Path
from typing import Optional, Callable


class FileReceiver:
    """文件接收器"""

    CHUNK_SIZE = 8192

    def __init__(self, port: int, save_dir: str):
        """
        初始化接收器

        Args:
            port: 监听端口
            save_dir: 文件保存目录
        """
        self.port = port
        self.save_dir = save_dir
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.running = False

    def start(self):
        """启动接收服务器"""
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(1)
        self.running = True

    def wait_for_file(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        timeout: float = 30.0
    ) -> Optional[str]:
        """
        等待接收文件

        Args:
            progress_callback: 进度回调 (received_bytes, total_bytes)
            timeout: 接收超时时间

        Returns:
            接收到的文件路径，或 None 表示失败
        """
        if not self.server_socket:
            print("服务器未启动")
            return None

        try:
            self.server_socket.settimeout(timeout)

            # 接受连接
            self.client_socket, addr = self.server_socket.accept()
            print(f"连接来自: {addr}")

            # 发送确认
            hostname = os.environ.get('COMPUTERNAME', 'Unknown')
            self.client_socket.send(f"ACK:{hostname}".encode())

            # 接收文件头
            header_data = self.client_socket.recv(1024).decode()
            header = json.loads(header_data)

            if header.get("type") != "FILE_HEADER":
                print(f"无效的文件头: {header}")
                return None

            filename = header["filename"]
            filesize = header["size"]

            # 发送准备就绪
            self.client_socket.send(b"READY")

            # 接收文件数据
            filepath = os.path.join(self.save_dir, filename)
            received = 0

            with open(filepath, 'wb') as f:
                while received < filesize:
                    chunk = self.client_socket.recv(self.CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    received += len(chunk)

                    if progress_callback:
                        progress_callback(received, filesize)

            # 发送确认
            self.client_socket.send(f"ACK:{received}".encode())

            # 等待完成信号
            done = self.client_socket.recv(1024).decode()
            if done == "DONE":
                self.client_socket.send(b"CONFIRM")
                return filepath

            return None

        except socket.timeout:
            print("接收超时")
            return None
        except Exception as e:
            print(f"接收失败: {e}")
            return None
        finally:
            self._close_client()

    def stop(self):
        """停止接收服务器"""
        self.running = False
        self._close_client()
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None

    def _close_client(self):
        """关闭客户端连接"""
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
