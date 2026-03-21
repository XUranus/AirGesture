#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         recorder.py
#*   Author:       XUranus
#*   Date:         2026-03-16
#*   Description:  
#*
#================================================================*/

"""gesture_capture/recorder.py — Video clip writer and CSV dataset manager."""

from __future__ import annotations

import csv
import os
import re
import time
from datetime import datetime
from typing import Optional

import cv2


class VideoRecorder:
    """Records individual video clips and appends metadata rows to a CSV file.

    Directory layout created by this class::

        <video_dir>/
            Grab_20240615_143022_123.avi
            Swipe_Left_20240615_143030_456.avi
            ...
        <csv_path>   (labels.csv)
    """

    CSV_FIELDS = [
        "filename",
        "label",
        "duration_sec",
        "num_frames",
        "fps",
        "width",
        "height",
        "timestamp",
    ]

    def __init__(self, video_dir: str, csv_path: str):
        self.video_dir = video_dir
        self.csv_path = csv_path
        os.makedirs(video_dir, exist_ok=True)

        self._writer: Optional[cv2.VideoWriter] = None
        self._label = ""
        self._filename = ""
        self._frame_count = 0
        self._start_time = 0.0
        self._fps = 30.0
        self._frame_size = (640, 480)
        self._recording = False

    # --------------------------------------------------------- public API

    def start(self, label: str, fps: float, frame_size: tuple):
        """Begin recording a new clip.

        Parameters
        ----------
        label : str
            The gesture label for this clip.
        fps : float
            Frames-per-second to write into the video container.
        frame_size : tuple of (width, height)
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe = re.sub(r"[^\w\-]", "_", label)
        self._filename = f"{safe}_{ts}.avi"
        filepath = os.path.join(self.video_dir, self._filename)

        self._label = label
        self._fps = fps
        self._frame_size = frame_size
        self._frame_count = 0
        self._start_time = time.time()

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writer = cv2.VideoWriter(filepath, fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot create VideoWriter for {filepath}")

        self._recording = True

    def write_frame(self, frame):
        """Append a BGR frame to the current clip."""
        if self._recording and self._writer is not None:
            self._writer.write(frame)
            self._frame_count += 1

    def stop(self) -> Optional[dict]:
        """Finish the current clip, save to disk, append row to CSV.

        Returns the metadata dict for the clip, or *None* if not recording.
        """
        if not self._recording:
            return None

        duration = time.time() - self._start_time
        self._writer.release()
        self._writer = None
        self._recording = False

        record = {
            "filename": self._filename,
            "label": self._label,
            "duration_sec": round(duration, 3),
            "num_frames": self._frame_count,
            "fps": round(self._fps, 1),
            "width": self._frame_size[0],
            "height": self._frame_size[1],
            "timestamp": datetime.now().isoformat(),
        }
        self._append_csv(record)
        return record

    # ----------------------------------------------------------- queries

    def get_label_counts(self) -> dict:
        """Return ``{label: count}`` from the existing CSV."""
        counts: dict[str, int] = {}
        if not os.path.exists(self.csv_path):
            return counts
        try:
            with open(self.csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    lbl = row.get("label", "")
                    counts[lbl] = counts.get(lbl, 0) + 1
        except Exception:
            pass
        return counts

    def get_total_clips(self) -> int:
        return sum(self.get_label_counts().values())

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ---------------------------------------------------------- internal

    def _append_csv(self, record: dict):
        write_header = (
            not os.path.exists(self.csv_path)
            or os.path.getsize(self.csv_path) == 0
        )
        with open(self.csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(record)

