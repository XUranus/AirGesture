"""
图片查看器 - 自动打开接收到的图片
"""

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional


class ImageViewer:
    """跨平台图片查看器"""

    @staticmethod
    def open_image(image_path: str) -> bool:
        """
        使用系统默认图片查看器打开图片

        Args:
            image_path: 图片文件路径

        Returns:
            是否成功打开
        """
        path = Path(image_path).absolute()

        if not path.exists():
            print(f"图片不存在: {path}")
            return False

        system = platform.system()

        try:
            if system == "Windows":
                # Windows: 使用默认图片查看器
                os.startfile(str(path))
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(path)])
            elif system == "Linux":
                subprocess.run(["xdg-open", str(path)])
            else:
                print(f"不支持的操作系统: {system}")
                return False

            return True
        except Exception as e:
            print(f"打开图片失败: {e}")
            return False

    @staticmethod
    def open_with_preview(image_path: str, save_dir: Optional[str] = None):
        """
        使用自定义预览窗口打开图片，提供保存/删除选项

        Args:
            image_path: 图片文件路径
            save_dir: 永久保存目录
        """
        import tkinter as tk
        from tkinter import messagebox
        from PIL import Image, ImageTk
        import shutil

        path = Path(image_path)
        save_dir = save_dir or str(Path.home() / "Pictures" / "AirGesture")

        root = tk.Tk()
        root.title(f"接收到的图片 - {path.name}")
        root.attributes('-topmost', True)

        # 加载图片
        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"加载图片失败: {e}")
            return

        # 计算合适的窗口大小
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        max_width = int(screen_width * 0.8)
        max_height = int(screen_height * 0.8)

        if img.width > max_width or img.height > max_height:
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(img)

        # 显示图片
        label = tk.Label(root, image=photo)
        label.image = photo
        label.pack()

        # 按钮框架
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        def save_permanently():
            """保存到永久目录"""
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            dest = save_path / path.name

            # 如果文件已存在，添加序号
            if dest.exists():
                base = path.stem
                ext = path.suffix
                i = 1
                while dest.exists():
                    dest = save_path / f"{base}_{i}{ext}"
                    i += 1

            shutil.copy(image_path, dest)
            messagebox.showinfo("保存成功", f"图片已保存到:\n{dest}")

        def delete_temp():
            """删除临时文件"""
            path.unlink(missing_ok=True)
            root.destroy()

        def close_only():
            """仅关闭，保留临时文件"""
            root.destroy()

        tk.Button(btn_frame, text="保存", command=save_permanently, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="删除", command=delete_temp, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="关闭", command=close_only, width=10).pack(side=tk.LEFT, padx=5)

        # 窗口居中
        root.update_idletasks()
        x = (screen_width - root.winfo_width()) // 2
        y = (screen_height - root.winfo_height()) // 2
        root.geometry(f"+{x}+{y}")

        root.mainloop()
