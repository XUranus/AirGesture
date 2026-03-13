"""
悬浮缩略图窗口 - 类似华为隔空取物效果
"""

import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import queue
from typing import Optional, Tuple


class ThumbnailWindow:
    """
    悬浮缩略图窗口

    截图后在屏幕一角显示缩略图，带有发送进度
    """

    def __init__(
        self,
        position: str = "bottom-right",
        size: Tuple[int, int] = (300, 200)
    ):
        """
        初始化缩略图窗口

        Args:
            position: 窗口位置 ("bottom-right", "bottom-left", "top-right", "top-left")
            size: 缩略图大小 (width, height)
        """
        self.position = position
        self.size = size
        self.root: Optional[tk.Tk] = None
        self.label: Optional[tk.Label] = None
        self.progress_label: Optional[tk.Label] = None
        self._closed = False
        self._message_queue = queue.Queue()  # 用于线程间通信
        self._thread = None

    def show(self, image_path: str, on_complete_callback=None):
        """
        显示缩略图

        Args:
            image_path: 截图文件路径
            on_complete_callback: 发送完成后的回调
        """
        # 在独立线程中运行 Tkinter
        self._thread = threading.Thread(
            target=self._show_window,
            args=(image_path, on_complete_callback),
            daemon=True
        )
        self._thread.start()

    def _show_window(self, image_path: str, on_complete_callback):
        """创建并显示窗口"""
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # 无边框窗口
        self.root.attributes('-topmost', True)  # 置顶
        self.root.attributes('-alpha', 0.9)  # 半透明

        # 计算位置
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        margin = 20

        if self.position == "bottom-right":
            x = screen_width - self.size[0] - margin
            y = screen_height - self.size[1] - margin - 60  # 留出任务栏空间
        elif self.position == "bottom-left":
            x = margin
            y = screen_height - self.size[1] - margin - 60
        elif self.position == "top-right":
            x = screen_width - self.size[0] - margin
            y = margin
        else:  # top-left
            x = margin
            y = margin

        self.root.geometry(f"{self.size[0]}x{self.size[1] + 30}+{x}+{y}")

        # 加载并缩放图片
        try:
            img = Image.open(image_path)
            img.thumbnail(self.size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            # 创建图片标签
            self.label = tk.Label(self.root, image=photo, bg='#2b2b2b')
            self.label.image = photo  # 保持引用
            self.label.pack()
        except Exception as e:
            print(f"加载图片失败: {e}")
            self.label = tk.Label(self.root, text="[截图预览]", bg='#2b2b2b', fg='white')
            self.label.pack()

        # 进度标签
        self.progress_label = tk.Label(
            self.root,
            text="准备发送...",
            fg='white',
            bg='#2b2b2b',
            font=('Arial', 10)
        )
        self.progress_label.pack()

        # 自动关闭定时器（用于发送完成后）
        self.root.after(10000, self._auto_close)

        # 定期检查消息队列
        self.root.after(50, self._process_messages)

        self.root.mainloop()

    def _process_messages(self):
        """处理来自其他线程的消息"""
        if self._closed or not self.root:
            return

        try:
            while True:
                msg = self._message_queue.get_nowait()
                msg_type = msg.get('type')
                if msg_type == 'update_progress':
                    self._do_update_progress(msg.get('percentage'), msg.get('status'))
                elif msg_type == 'show_success':
                    self._do_show_success()
                elif msg_type == 'show_error':
                    self._do_show_error(msg.get('message'))
                elif msg_type == 'close':
                    self._close()
                    return
        except queue.Empty:
            pass

        # 继续检查消息队列
        if not self._closed and self.root:
            self.root.after(50, self._process_messages)

    def update_progress(self, percentage: float, status: str = None):
        """
        更新进度（线程安全）

        Args:
            percentage: 进度百分比
            status: 状态文本
        """
        if self._closed:
            return
        self._message_queue.put({
            'type': 'update_progress',
            'percentage': percentage,
            'status': status
        })

    def _do_update_progress(self, percentage: float, status: str = None):
        """实际更新进度（在主线程中执行）"""
        if self._closed or not self.progress_label:
            return
        text = status or f"发送中 {percentage:.0f}%"
        try:
            self.progress_label.config(text=text)
        except:
            pass

    def show_success(self):
        """显示发送成功（线程安全）"""
        if self._closed:
            return
        self._message_queue.put({'type': 'show_success'})

    def _do_show_success(self):
        """实际显示成功（在主线程中执行）"""
        if self._closed or not self.progress_label:
            return
        try:
            self.progress_label.config(text="✓ 发送成功", fg='#4CAF50')
        except:
            pass
        if self.root:
            self.root.after(1500, self._fade_out)

    def show_error(self, message: str = "发送失败"):
        """显示发送失败（线程安全）"""
        if self._closed:
            return
        self._message_queue.put({'type': 'show_error', 'message': message})

    def _do_show_error(self, message: str = "发送失败"):
        """实际显示错误（在主线程中执行）"""
        if self._closed or not self.progress_label:
            return
        try:
            self.progress_label.config(text=f"✗ {message}", fg='#f44336')
        except:
            pass

    def _fade_out(self):
        """淡出动画"""
        if self._closed or not self.root:
            return

        alpha = 0.9
        for _ in range(10):
            alpha -= 0.09
            try:
                self.root.attributes('-alpha', max(0, alpha))
                self.root.update()
                time.sleep(0.05)
            except:
                break
        self._close()

    def _auto_close(self):
        """自动关闭"""
        self._close()

    def _close(self):
        """关闭窗口"""
        if self._closed:
            return

        self._closed = True
        if self.root:
            try:
                # 在 Tkinter 线程中安全关闭
                self.root.after(0, self._do_destroy)
            except:
                pass

    def _do_destroy(self):
        """实际销毁窗口（在 Tkinter 线程中执行）"""
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass
        self.root = None

    def is_closed(self) -> bool:
        """检查窗口是否已关闭"""
        return self._closed
