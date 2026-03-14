#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         overlay.py
#*   Author:       XUranus
#*   Date:         2026-03-14
#*   Description:  
#*
#================================================================*/

# /GrabDrop-Desktop/overlay.py
import logging
import platform
import threading
import time
from typing import Optional

logger = logging.getLogger("Overlay")


class OverlayManager:
    """
    Cross-platform overlay using Tkinter.
    Shows wakeup indicator, flash, thumbnail, ripple on the screen.
    """

    def __init__(self):
        self._tk = None
        self._root = None
        self._indicator_window = None
        self._flash_window = None
        self._started = False
        self._system = platform.system()

    def start(self):
        """Initialize Tk root on current (main) thread."""
        try:
            import tkinter as tk

            self._tk = tk
            self._root = tk.Tk()
            self._root.withdraw()
            self._root.title("GrabDrop")
            self._root.protocol("WM_DELETE_WINDOW", lambda: None)

            self._started = True
            logger.info("Overlay manager initialized (Tkinter)")
        except Exception as e:
            logger.error(f"Overlay init failed: {e}")
            self._started = False

    def is_available(self) -> bool:
        return self._started

    def get_root(self):
        return self._root

    def stop(self):
        if self._root:
            try:
                self._hide_indicator()
                self._root.quit()
                self._root.destroy()
            except Exception:
                pass
        self._started = False

    # --- Wakeup Indicator ---

    def show_wakeup_indicator(self, emoji: str):
        if not self._started:
            return
        try:
            self._root.after(0, lambda: self._show_indicator(emoji))
        except Exception:
            pass

    def hide_wakeup_indicator(self):
        if not self._started:
            return
        try:
            self._root.after(0, self._hide_indicator)
        except Exception:
            pass

    def _show_indicator(self, emoji: str):
        tk = self._tk
        if tk is None:
            return

        self._hide_indicator()

        try:
            win = tk.Toplevel(self._root)
            win.overrideredirect(True)
            win.attributes("-topmost", True)

            bg_color = "#1E1E2E"

            if self._system == "Windows":
                win.attributes("-transparentcolor", "black")
            elif self._system == "Darwin":
                try:
                    win.attributes("-transparent", True)
                    bg_color = "systemTransparent"
                except Exception:
                    pass
            else:
                # Linux
                try:
                    win.attributes("-alpha", 0.9)
                except Exception:
                    pass

            win.configure(bg=bg_color)

            frame = tk.Frame(win, bg="#1E1E2E", padx=16, pady=8)
            frame.pack()

            # Try different fonts for emoji
            emoji_fonts = [
                ("Noto Color Emoji", 32),
                ("Segoe UI Emoji", 32),
                ("Apple Color Emoji", 32),
                ("DejaVu Sans", 28),
                ("TkDefaultFont", 28),
            ]

            label = None
            for font_name, font_size in emoji_fonts:
                try:
                    label = tk.Label(
                        frame,
                        text=emoji,
                        font=(font_name, font_size),
                        bg="#1E1E2E",
                        fg="white",
                    )
                    label.pack()
                    break
                except Exception:
                    continue

            if label is None:
                label = tk.Label(
                    frame,
                    text=emoji,
                    font=("", 28),
                    bg="#1E1E2E",
                    fg="white",
                )
                label.pack()

            # Position: top center
            win.update_idletasks()
            win_width = win.winfo_reqwidth()
            screen_width = win.winfo_screenwidth()
            x = (screen_width - win_width) // 2
            y = 30
            win.geometry(f"+{x}+{y}")

            self._indicator_window = win
            self._pulse_indicator(win, label, 0)

            logger.debug(f"Wakeup indicator shown: {emoji}")
        except Exception as e:
            logger.error(f"Show indicator failed: {e}")

    def _pulse_indicator(self, win, label, step):
        if not self._indicator_window or win != self._indicator_window:
            return
        try:
            if not win.winfo_exists():
                return
            sizes = [28, 30, 32, 30, 28, 26, 24, 26]
            size = sizes[step % len(sizes)]
            current_font = label.cget("font")
            # Extract font family name
            font_family = current_font.split()[0] if current_font else ""
            label.configure(font=(font_family, size))
            win.after(
                150,
                lambda: self._pulse_indicator(win, label, step + 1),
            )
        except Exception:
            pass

    def _hide_indicator(self):
        if self._indicator_window:
            try:
                self._indicator_window.destroy()
            except Exception:
                pass
            self._indicator_window = None
            logger.debug("Wakeup indicator hidden")

    # --- Flash Effect ---

    def show_flash(self):
        if not self._started:
            return
        try:
            self._root.after(0, self._do_flash)
        except Exception:
            pass

    def _do_flash(self):
        tk = self._tk
        if tk is None:
            return

        try:
            win = tk.Toplevel(self._root)
            win.overrideredirect(True)
            win.attributes("-topmost", True)

            screen_w = win.winfo_screenwidth()
            screen_h = win.winfo_screenheight()
            win.geometry(f"{screen_w}x{screen_h}+0+0")
            win.configure(bg="white")

            try:
                win.attributes("-alpha", 0.7)
            except Exception:
                pass

            self._fade_flash(win, 0.7, 0)
            logger.debug("Flash shown")
        except Exception as e:
            logger.error(f"Flash failed: {e}")

    def _fade_flash(self, win, current_alpha, step):
        if step > 6:
            self._safe_destroy(win)
            return

        try:
            if not win.winfo_exists():
                return
            new_alpha = max(0, current_alpha - 0.12)
            win.attributes("-alpha", new_alpha)
            win.after(
                40,
                lambda: self._fade_flash(win, new_alpha, step + 1),
            )
        except Exception:
            self._safe_destroy(win)

    # --- Thumbnail ---

    def show_thumbnail(self, filepath: str):
        if not self._started:
            return
        try:
            self._root.after(0, lambda: self._do_thumbnail(filepath))
        except Exception:
            pass

    def _do_thumbnail(self, filepath: str):
        tk = self._tk
        if tk is None:
            return

        try:
            from PIL import Image, ImageTk

            img = Image.open(filepath)
            thumb_w = 200
            ratio = thumb_w / img.width
            thumb_h = int(img.height * ratio)
            img = img.resize((thumb_w, thumb_h), Image.LANCZOS)

            win = tk.Toplevel(self._root)
            win.overrideredirect(True)
            win.attributes("-topmost", True)

            photo = ImageTk.PhotoImage(img)

            frame = tk.Frame(win, bg="white", padx=3, pady=3)
            frame.pack()

            label = tk.Label(frame, image=photo, bg="white")
            label.image = photo  # prevent GC
            label.pack()

            # Bottom-left
            win.update_idletasks()
            screen_h = win.winfo_screenheight()
            x = 20
            y = screen_h - thumb_h - 80
            win.geometry(f"+{x}+{y}")

            try:
                win.attributes("-alpha", 1.0)
            except Exception:
                pass

            # Fade out after 2s
            win.after(2000, lambda: self._fade_destroy(win, 1.0, 0))
            logger.debug("Thumbnail shown")
        except Exception as e:
            logger.error(f"Thumbnail failed: {e}")

    # --- Ripple ---

    def show_ripple(self):
        if not self._started:
            return
        try:
            self._root.after(0, self._do_ripple)
        except Exception:
            pass

    def _do_ripple(self):
        tk = self._tk
        if tk is None:
            return

        try:
            win = tk.Toplevel(self._root)
            win.overrideredirect(True)
            win.attributes("-topmost", True)

            screen_w = win.winfo_screenwidth()
            screen_h = win.winfo_screenheight()
            win.geometry(f"{screen_w}x{screen_h}+0+0")

            try:
                win.attributes("-alpha", 0.5)
            except Exception:
                pass

            canvas_bg = "black"
            canvas = tk.Canvas(
                win,
                width=screen_w,
                height=screen_h,
                bg=canvas_bg,
                highlightthickness=0,
            )
            canvas.pack()

            cx = screen_w // 2
            cy = screen_h // 2

            self._animate_ripple(win, canvas, cx, cy, 0, 20)
            logger.debug("Ripple shown")
        except Exception as e:
            logger.error(f"Ripple failed: {e}")

    def _animate_ripple(self, win, canvas, cx, cy, step, max_steps):
        if step >= max_steps:
            self._safe_destroy(win)
            return

        try:
            if not win.winfo_exists():
                return

            canvas.delete("ripple")
            fraction = step / max_steps
            max_r = max(
                canvas.winfo_width(), canvas.winfo_height()
            ) // 2
            if max_r <= 0:
                max_r = 800
            r = int(max_r * fraction)
            width1 = 3 + int(15 * fraction)

            canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline="#42A5F5",
                width=width1,
                tags="ripple",
            )

            r2 = max(0, r - 60)
            if r2 > 0:
                canvas.create_oval(
                    cx - r2, cy - r2, cx + r2, cy + r2,
                    outline="#81D4FA",
                    width=max(1, width1 // 2),
                    tags="ripple",
                )

            canvas.after(
                30,
                lambda: self._animate_ripple(
                    win, canvas, cx, cy, step + 1, max_steps
                ),
            )
        except Exception:
            self._safe_destroy(win)

    # --- Helpers ---

    def _fade_destroy(self, win, alpha, step):
        if step > 5:
            self._safe_destroy(win)
            return
        try:
            if not win.winfo_exists():
                return
            new_alpha = max(0, alpha - 0.2)
            win.attributes("-alpha", new_alpha)
            win.after(
                50,
                lambda: self._fade_destroy(win, new_alpha, step + 1),
            )
        except Exception:
            self._safe_destroy(win)

    def _safe_destroy(self, win):
        try:
            if win and win.winfo_exists():
                win.destroy()
        except Exception:
            pass

