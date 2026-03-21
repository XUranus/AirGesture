#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         main_window.py
#*   Author:       XUranus
#*   Date:         2026-03-16
#*   Description:  
#*
#================================================================*/

"""gesture_capture/main_window.py — Main application window."""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from camera_thread import CameraThread
from recorder import VideoRecorder

# ──────────────────────────────────────────────────────────── Dark theme ──

_DARK_STYLE = """
QMainWindow { background-color: #2b2b2b; }
QGroupBox {
    color: #ddd; font-weight: bold;
    border: 1px solid #555; border-radius: 6px;
    margin-top: 10px; padding-top: 18px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
}
QLabel        { color: #ccc; }
QLineEdit {
    background-color: #3c3c3c; color: #eee;
    border: 1px solid #555; border-radius: 4px;
    padding: 6px 10px; font-size: 14px;
}
QLineEdit:focus { border: 1px solid #3498db; }
QLineEdit:disabled {
    background-color: #333; color: #777;
}
QPushButton {
    background-color: #3c3c3c; color: #ddd;
    border: 1px solid #555; border-radius: 4px;
    padding: 6px 16px; font-size: 12px;
}
QPushButton:hover { background-color: #4a4a4a; }
QStatusBar { background-color: #1e1e1e; color: #aaa; }
"""

_REC_BTN_IDLE = """
QPushButton {
    background-color: #c0392b; color: white;
    border: none; border-radius: 8px;
}
QPushButton:hover { background-color: #e74c3c; }
QPushButton:pressed { background-color: #a93226; }
"""

_REC_BTN_ACTIVE = """
QPushButton {
    background-color: #27ae60; color: white;
    border: none; border-radius: 8px;
}
QPushButton:hover { background-color: #2ecc71; }
QPushButton:pressed { background-color: #1e8449; }
"""

# ─────────────────────────────────────────────────────────── Main window ──


class MainWindow(QMainWindow):
    def __init__(self, camera_index: int = 0, output_dir: str | None = None):
        super().__init__()
        self.setWindowTitle("Gesture Motion Capture")
        self.setMinimumSize(820, 720)
        self.setStyleSheet(_DARK_STYLE)

        # ---- state ----
        self.current_frame: np.ndarray | None = None
        self.camera_fps = 30.0
        self.camera_resolution = (640, 480)
        self.recording = False
        self.record_start_time = 0.0

        # ---- paths ----
        self.output_dir = output_dir or os.path.join(os.getcwd(), "dataset")
        self.video_dir = os.path.join(self.output_dir, "videos")
        self.csv_path = os.path.join(self.output_dir, "labels.csv")
        os.makedirs(self.video_dir, exist_ok=True)

        # ---- recorder ----
        self.recorder = VideoRecorder(self.video_dir, self.csv_path)
        self.label_counts: dict[str, int] = self.recorder.get_label_counts()
        self.total_clips: int = sum(self.label_counts.values())

        # ---- build UI ----
        self._build_ui()
        self._setup_shortcuts()

        # ---- FPS tracking ----
        self._fps_counter = 0
        self._fps_time = time.time()

        # ---- recording elapsed timer ----
        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(50)
        self._rec_timer.timeout.connect(self._update_rec_timer)

        # ---- camera thread (start last) ----
        self._camera = CameraThread(camera_index=camera_index)
        self._camera.frame_ready.connect(self._on_frame)
        self._camera.camera_opened.connect(self._on_camera_opened)
        self._camera.error_occurred.connect(self._on_camera_error)
        self._camera.start()

    # ================================================================== #
    #                             UI SETUP                                #
    # ================================================================== #

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(10)

        # ---- camera preview ----
        self.preview = QLabel("Initializing camera …")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 480)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setStyleSheet(
            "background-color: #1a1a2e; color: #666;"
            "border: 2px solid #333; border-radius: 8px; font-size: 16px;"
        )
        root.addWidget(self.preview, stretch=1)

        # ---- controls group ----
        grp = QGroupBox("Capture Controls")
        grid = QGridLayout(grp)
        grid.setSpacing(8)

        # row 0 — label
        grid.addWidget(QLabel("Gesture Label:"), 0, 0)
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText(
            "e.g.  Grab,  Swipe_Left,  Pinch,  Wave …"
        )
        self.label_input.setMinimumHeight(34)
        self.label_input.textChanged.connect(self._on_label_text_changed)
        grid.addWidget(self.label_input, 0, 1)

        self.label_count_lbl = QLabel("")
        self.label_count_lbl.setFixedWidth(100)
        self.label_count_lbl.setAlignment(Qt.AlignCenter)
        self.label_count_lbl.setStyleSheet("color: #888;")
        grid.addWidget(self.label_count_lbl, 0, 2)

        # row 1 — output dir
        grid.addWidget(QLabel("Output Dir:"), 1, 0)
        self.dir_display = QLabel(self._short_path(self.output_dir))
        self.dir_display.setToolTip(self.output_dir)
        self.dir_display.setStyleSheet("color: #777;")
        grid.addWidget(self.dir_display, 1, 1)
        browse_btn = QPushButton("Browse …")
        browse_btn.setFixedWidth(100)
        browse_btn.clicked.connect(self._browse_output_dir)
        grid.addWidget(browse_btn, 1, 2)

        # row 2 — big record button
        self.rec_btn = QPushButton("⏺  START RECORDING    (Ctrl+R)")
        self.rec_btn.setMinimumHeight(54)
        self.rec_btn.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.rec_btn.setStyleSheet(_REC_BTN_IDLE)
        self.rec_btn.clicked.connect(self._toggle_recording)
        grid.addWidget(self.rec_btn, 2, 0, 1, 3)

        root.addWidget(grp)

        # ---- status bar ----
        sb = QStatusBar()
        self.setStatusBar(sb)

        self.status_msg = QLabel("Ready — press Ctrl+R or click the button to record")
        self.rec_time_lbl = QLabel("")
        self.clips_lbl = QLabel(f"Total clips: {self.total_clips}")
        self.fps_lbl = QLabel("FPS: —")

        sb.addWidget(self.status_msg, 1)
        sb.addPermanentWidget(self.rec_time_lbl)
        sb.addPermanentWidget(self.clips_lbl)
        sb.addPermanentWidget(self.fps_lbl)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+R"), self, self._toggle_recording)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

    # ================================================================== #
    #                            CAMERA SLOTS                             #
    # ================================================================== #

    def _on_camera_opened(self, info: dict):
        self.camera_fps = info["fps"]
        self.camera_resolution = (info["width"], info["height"])
        self.status_msg.setText(
            f"Camera ready — {info['width']}×{info['height']} @ {info['fps']:.0f} fps"
        )

    def _on_frame(self, frame: np.ndarray):
        self.current_frame = frame

        # ---- fps display ----
        self._fps_counter += 1
        dt = time.time() - self._fps_time
        if dt >= 1.0:
            self.fps_lbl.setText(f"FPS: {self._fps_counter / dt:.0f}")
            self._fps_counter = 0
            self._fps_time = time.time()

        # ---- write to recorder ----
        if self.recording:
            self.recorder.write_frame(frame)

        # ---- display (with optional overlay) ----
        display = frame.copy()
        if self.recording:
            self._draw_recording_overlay(display)
        self._render_preview(display)

    def _on_camera_error(self, msg: str):
        self.preview.setText(f"⚠  {msg}")
        QMessageBox.critical(self, "Camera Error", msg)

    # ================================================================== #
    #                           RECORDING                                 #
    # ================================================================== #

    def _toggle_recording(self):
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        label = self.label_input.text().strip()
        if not label:
            QMessageBox.warning(
                self, "Label Required",
                "Please type a gesture label before recording."
            )
            self.label_input.setFocus()
            return
        if self.current_frame is None:
            QMessageBox.warning(self, "No Camera", "The camera is not ready yet.")
            return

        h, w = self.current_frame.shape[:2]
        try:
            self.recorder.start(label, self.camera_fps, (w, h))
        except RuntimeError as exc:
            QMessageBox.critical(self, "Recorder Error", str(exc))
            return

        self.recording = True
        self.record_start_time = time.time()

        self.rec_btn.setText("⏹  STOP RECORDING    (Ctrl+R)")
        self.rec_btn.setStyleSheet(_REC_BTN_ACTIVE)
        self.label_input.setEnabled(False)
        self.status_msg.setText(f"🔴  Recording gesture: {label}")
        self._rec_timer.start()

    def _stop_recording(self):
        self.recording = False
        self._rec_timer.stop()

        record = self.recorder.stop()

        if record:
            self.total_clips += 1
            lbl = record["label"]
            self.label_counts[lbl] = self.label_counts.get(lbl, 0) + 1
            self.clips_lbl.setText(f"Total clips: {self.total_clips}")
            self._on_label_text_changed(self.label_input.text())
            self.status_msg.setText(
                f"✓  Saved {record['filename']}  —  "
                f"{record['duration_sec']}s · {record['num_frames']} frames"
            )

        self.rec_btn.setText("⏺  START RECORDING    (Ctrl+R)")
        self.rec_btn.setStyleSheet(_REC_BTN_IDLE)
        self.label_input.setEnabled(True)
        self.rec_time_lbl.setText("")

    # ================================================================== #
    #                             HELPERS                                 #
    # ================================================================== #

    def _draw_recording_overlay(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        elapsed = time.time() - self.record_start_time

        # Red border
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

        # Blinking dot + REC text
        if int(elapsed * 3) % 2 == 0:
            cv2.circle(frame, (28, 32), 10, (0, 0, 255), -1)
        cv2.putText(
            frame, f"REC  {elapsed:.1f}s", (48, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA,
        )

        # Label overlay
        label = self.label_input.text().strip()
        if label:
            cv2.putText(
                frame, label, (28, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )

    def _render_preview(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview.setPixmap(scaled)

    def _update_rec_timer(self):
        if self.recording:
            el = time.time() - self.record_start_time
            self.rec_time_lbl.setText(f"⏱  {el:.1f}s")

    def _on_label_text_changed(self, text: str):
        text = text.strip()
        if text and text in self.label_counts:
            n = self.label_counts[text]
            self.label_count_lbl.setText(f"({n} clip{'s' if n != 1 else ''})")
        else:
            self.label_count_lbl.setText("")

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose Dataset Output Directory", self.output_dir
        )
        if not path:
            return
        self.output_dir = path
        self.video_dir = os.path.join(path, "videos")
        self.csv_path = os.path.join(path, "labels.csv")
        os.makedirs(self.video_dir, exist_ok=True)

        self.recorder = VideoRecorder(self.video_dir, self.csv_path)
        self.label_counts = self.recorder.get_label_counts()
        self.total_clips = sum(self.label_counts.values())
        self.clips_lbl.setText(f"Total clips: {self.total_clips}")
        self.dir_display.setText(self._short_path(path))
        self.dir_display.setToolTip(path)
        self._on_label_text_changed(self.label_input.text())

    @staticmethod
    def _short_path(p: str, max_len: int = 50) -> str:
        return p if len(p) <= max_len else "…" + p[-(max_len - 1):]

    # ================================================================== #
    #                            CLEANUP                                  #
    # ================================================================== #

    def closeEvent(self, event):
        if self.recording:
            self._stop_recording()
        self._camera.stop()
        event.accept()

