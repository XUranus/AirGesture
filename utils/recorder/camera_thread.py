#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         camera_thread.py
#*   Author:       XUranus
#*   Date:         2026-03-16
#*   Description:  
#*
#================================================================*/

"""gesture_capture/camera_thread.py — Threaded camera capture using OpenCV."""

from __future__ import annotations

import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class CameraThread(QThread):
    """Continuously reads frames from a camera in a background thread.

    Signals
    -------
    frame_ready(np.ndarray)
        Emitted for every captured frame (BGR, already horizontally flipped).
    camera_opened(dict)
        Emitted once after the camera is successfully opened.
        Payload: {"width": int, "height": int, "fps": float}
    error_occurred(str)
        Emitted when the camera cannot be opened or a read fails.
    """

    frame_ready = pyqtSignal(np.ndarray)
    camera_opened = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False

    # ------------------------------------------------------------------ #

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error_occurred.emit(
                f"Cannot open camera index {self.camera_index}.\n"
                "Make sure the camera is connected and not used by another app."
            )
            return

        # Try to configure resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Read back actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0

        self.camera_opened.emit({"width": width, "height": height, "fps": fps})

        self._running = True
        frame_interval = 1.0 / fps

        while self._running:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                self.error_occurred.emit("Failed to read a frame from the camera.")
                break

            # Mirror for a natural front-camera feel
            frame = cv2.flip(frame, 1)
            self.frame_ready.emit(frame.copy())

            # Throttle to target FPS
            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()

    def stop(self):
        """Signal the thread to stop and wait for it to finish."""
        self._running = False
        self.wait(3000)

