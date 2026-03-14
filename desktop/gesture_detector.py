#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         gesture_detector.py
#*   Author:       XUranus
#*   Date:         2026-03-14
#*   Description:  
#*
#================================================================*/

# /GrabDrop-Desktop/gesture_detector.py
import logging
import time
import threading
from collections import deque
from enum import Enum
from typing import Callable, Optional

import cv2
import numpy as np

import config
from hand_landmark import HandLandmarkDetector, HandState, DetectionDetail

logger = logging.getLogger("GestureDetector")


class GestureEvent(Enum):
    GRAB = "GRAB"
    RELEASE = "RELEASE"


class Stage(Enum):
    IDLE = "IDLE"
    WAKEUP = "WAKEUP"


class GestureDetector:
    def __init__(self):
        self.on_gesture: Optional[Callable[[GestureEvent], None]] = None
        self.on_stage_change: Optional[Callable[[str, str], None]] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._detector: Optional[HandLandmarkDetector] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # Stage
        self._stage = Stage.IDLE

        # Idle state
        self._idle_window: deque = deque(maxlen=config.IDLE_WINDOW_SIZE)
        self._idle_frame_count = 0

        # Wakeup state
        self._wakeup_start_time = 0.0
        self._wakeup_trigger_state = HandState.NONE
        self._wakeup_target_state = HandState.NONE
        self._consecutive_target = 0
        self._wakeup_frames: list = []
        self._wakeup_frame_count = 0

        # Cooldown
        self._last_gesture_time = 0.0

        # Stats
        self._total_frames = 0
        self._camera_frames = 0
        self._last_stats_time = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._detector:
            self._detector.close()
            self._detector = None
        logger.info("Gesture detector stopped")

    def _run(self):
        logger.info("Initializing camera and MediaPipe...")

        # Initialize detector
        self._detector = HandLandmarkDetector()
        if not self._detector.is_initialized:
            logger.error("HandLandmarkDetector failed to init — aborting")
            return

        # Initialize camera
        self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self._cap.isOpened():
            logger.error(
                f"Failed to open camera index {config.CAMERA_INDEX}"
            )
            # Try other indices
            for idx in range(4):
                if idx == config.CAMERA_INDEX:
                    continue
                logger.info(f"Trying camera index {idx}...")
                self._cap = cv2.VideoCapture(idx)
                if self._cap.isOpened():
                    logger.info(f"Camera {idx} opened!")
                    break
            else:
                logger.error("No camera found!")
                return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"📷 Camera opened: {actual_w}x{actual_h} @ {fps:.0f}fps"
        )
        logger.info(
            f"👁️ IDLE stage — scanning for hand at ~{config.IDLE_FPS}fps"
        )

        self._last_stats_time = time.time()
        last_frame_time = 0.0

        while self._running:
            now = time.time()

            # Throttle
            min_interval = (
                config.IDLE_FRAME_INTERVAL_S
                if self._stage == Stage.IDLE
                else config.WAKEUP_FRAME_INTERVAL_S
            )
            elapsed = now - last_frame_time
            if elapsed < min_interval:
                # Sleep a bit to avoid busy waiting
                time.sleep(max(0, min_interval - elapsed - 0.002))
                continue

            ret, frame = self._cap.read()
            self._camera_frames += 1

            if not ret or frame is None:
                logger.warning("Camera read failed")
                time.sleep(0.1)
                continue

            last_frame_time = time.time()
            self._total_frames += 1

            # Mirror horizontally (webcam is usually mirrored)
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect
            detail = self._detector.detect(frame_rgb)

            # Route to stage handler
            if self._stage == Stage.IDLE:
                self._process_idle(detail)
            else:
                self._process_wakeup(detail)

            # Periodic stats
            if now - self._last_stats_time > 10:
                self._last_stats_time = now
                logger.info(
                    f"📊 Stats: camera={self._camera_frames} "
                    f"processed={self._total_frames} "
                    f"stage={self._stage.value}"
                )

        # Cleanup
        if self._cap:
            self._cap.release()
        if self._detector:
            self._detector.close()

    # --- IDLE ---

    def _process_idle(self, detail: DetectionDetail):
        self._idle_frame_count += 1
        self._idle_window.append(detail.state)

        # Log every 10 frames
        if self._idle_frame_count % 10 == 0:
            window_str = "".join(self._state_emoji(s) for s in self._idle_window)
            palm_n = sum(1 for s in self._idle_window if s == HandState.PALM)
            fist_n = sum(1 for s in self._idle_window if s == HandState.FIST)
            none_n = sum(1 for s in self._idle_window if s == HandState.NONE)
            unk_n = sum(1 for s in self._idle_window if s == HandState.UNKNOWN)

            logger.info(
                f"👁️ IDLE #{self._idle_frame_count} | {window_str} | "
                f"P={palm_n} F={fist_n} N={none_n} U={unk_n} | "
                f"det: {detail.summary()}"
            )

        if len(self._idle_window) < config.IDLE_WINDOW_SIZE:
            return

        palm_count = sum(
            1 for s in self._idle_window if s == HandState.PALM
        )
        fist_count = sum(
            1 for s in self._idle_window if s == HandState.FIST
        )

        if palm_count >= config.IDLE_TRIGGER_THRESHOLD:
            self._enter_wakeup(HandState.PALM, HandState.FIST)
        elif fist_count >= config.IDLE_TRIGGER_THRESHOLD:
            self._enter_wakeup(HandState.FIST, HandState.PALM)

    # --- WAKEUP ---

    def _enter_wakeup(
        self, trigger: HandState, target: HandState
    ):
        now = time.time()
        if now - self._last_gesture_time < config.GRAB_COOLDOWN_S:
            logger.info("⏳ Wakeup suppressed — cooldown active")
            return

        self._stage = Stage.WAKEUP
        self._wakeup_start_time = time.time()
        self._wakeup_trigger_state = trigger
        self._wakeup_target_state = target
        self._consecutive_target = 0
        self._wakeup_frames = []
        self._wakeup_frame_count = 0
        self._idle_window.clear()
        self._idle_frame_count = 0

        trigger_e = "🖐" if trigger == HandState.PALM else "✊"
        target_e = "✊" if target == HandState.FIST else "🖐"
        motion = "GRAB" if target == HandState.FIST else "RELEASE"

        logger.info(
            f"🔔 WAKEUP! Detected {trigger_e} — "
            f"watching 2s for {trigger_e}→{target_e} ({motion})"
        )

        # Notify overlay
        indicator = "✊" if target == HandState.FIST else "🤚"
        if self.on_stage_change:
            self.on_stage_change("WAKEUP", indicator)

    def _process_wakeup(self, detail: DetectionDetail):
        self._wakeup_frame_count += 1
        self._wakeup_frames.append(detail.state)

        elapsed = time.time() - self._wakeup_start_time
        remaining = config.WAKEUP_DURATION_S - elapsed

        # Track consecutive target frames
        if detail.state == self._wakeup_target_state:
            self._consecutive_target += 1
        else:
            if self._consecutive_target > 0:
                logger.debug(
                    f"Streak broken at {self._consecutive_target} "
                    f"(got {detail.state}, need {self._wakeup_target_state})"
                )
            self._consecutive_target = 0

        # Log
        if (
            self._wakeup_frame_count % 5 == 0
            or self._consecutive_target > 0
        ):
            state_e = self._state_emoji(detail.state)
            progress = self._progress_bar(
                self._consecutive_target, config.WAKEUP_CONFIRM_FRAMES
            )
            logger.info(
                f"⏱️ WK #{self._wakeup_frame_count} "
                f"{remaining:.1f}s | {state_e} "
                f"str={self._consecutive_target}/"
                f"{config.WAKEUP_CONFIRM_FRAMES} "
                f"{progress} | {detail.summary()}"
            )

        # Timeout
        if elapsed > config.WAKEUP_DURATION_S:
            summary = self._wakeup_summary()
            logger.info(f"⌛ WAKEUP timeout — no gesture. {summary}")
            self._exit_wakeup()
            return

        # Confirm
        if self._consecutive_target >= config.WAKEUP_CONFIRM_FRAMES:
            if self._wakeup_target_state == HandState.FIST:
                logger.info("✅ ✊ GRAB CONFIRMED! (palm→fist)")
                event = GestureEvent.GRAB
            else:
                logger.info("✅ 🖐 RELEASE CONFIRMED! (fist→palm)")
                event = GestureEvent.RELEASE

            self._last_gesture_time = time.time()
            self._exit_wakeup()

            if self.on_gesture:
                # Run handler in a separate thread to not block camera
                threading.Thread(
                    target=self.on_gesture,
                    args=(event,),
                    daemon=True,
                ).start()

    def _exit_wakeup(self):
        self._stage = Stage.IDLE
        self._consecutive_target = 0
        self._wakeup_frames = []
        self._wakeup_frame_count = 0
        self._idle_window.clear()
        self._idle_frame_count = 0

        if self.on_stage_change:
            self.on_stage_change("IDLE", "")

        logger.info("👁️ Back to IDLE — scanning at ~10fps")

    # --- Helpers ---

    @staticmethod
    def _state_emoji(state: HandState) -> str:
        return {
            HandState.PALM: "🖐",
            HandState.FIST: "✊",
            HandState.UNKNOWN: "❓",
            HandState.NONE: "·",
        }.get(state, "?")

    @staticmethod
    def _progress_bar(current: int, maximum: int) -> str:
        filled = min(current, maximum)
        empty = maximum - filled
        return "[" + "█" * filled + "░" * empty + "]"

    def _wakeup_summary(self) -> str:
        total = len(self._wakeup_frames)
        palm = sum(1 for s in self._wakeup_frames if s == HandState.PALM)
        fist = sum(1 for s in self._wakeup_frames if s == HandState.FIST)
        none = sum(1 for s in self._wakeup_frames if s == HandState.NONE)
        unk = sum(1 for s in self._wakeup_frames if s == HandState.UNKNOWN)

        timeline = "".join(
            GestureDetector._state_emoji(s)
            for s in self._wakeup_frames[-30:]
        )
        return (
            f"total={total} 🖐={palm} ✊={fist} ·={none} ❓={unk} "
            f"| {timeline}"
        )

