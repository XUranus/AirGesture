#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         gesture_detector.py
#*   Author:       XUranus
#*   Date:         2026-03-14
#*   Description:  Dual-mode gesture detector (Neural Network + Legacy).
#*
#================================================================*/

"""
Real-time gesture detector supporting two detection backends:

- **Neural Network (TCN)**: Uses a Temporal Convolutional Network via ONNX Runtime
  for 5-class classification (grab/release/swipe_up/swipe_down/noise).

- **Legacy (rule-based)**: Uses hand landmark ratios for palm/fist classification
  and frame-to-frame velocity for swipe detection.

The user selects the preferred mode via ``config.DETECTION_METHOD``.
If TCN mode is selected but the model fails to load, the detector
automatically falls back to legacy mode.
"""

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
from gesture_classifier import GestureClassifier

logger = logging.getLogger("GestureDetector")


class GestureEvent(Enum):
    GRAB = "GRAB"
    RELEASE = "RELEASE"
    SWIPE_UP = "SWIPE_UP"
    SWIPE_DOWN = "SWIPE_DOWN"


class Stage(Enum):
    IDLE = "IDLE"
    WAKEUP = "WAKEUP"


class DetectionMode(Enum):
    NEURAL_NETWORK = "neural_network"
    LEGACY = "legacy"


class GestureDetector:
    def __init__(self):
        self.on_gesture: Optional[Callable[[GestureEvent], None]] = None
        self.on_stage_change: Optional[Callable[[str, str], None]] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._detector: Optional[HandLandmarkDetector] = None
        self._classifier: Optional[GestureClassifier] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # Active detection mode (resolved at start)
        self._active_mode = DetectionMode.LEGACY

        # Stage
        self._stage = Stage.IDLE

        # ── Shared IDLE state ─────────────────────────────────────
        self._idle_window_states: deque = deque(maxlen=config.IDLE_WINDOW_SIZE)  # legacy
        self._idle_window_hand: deque = deque(maxlen=config.IDLE_WINDOW_SIZE)    # NN
        self._idle_frame_count = 0

        # ── Shared WAKEUP state ───────────────────────────────────
        self._wakeup_start_time = 0.0
        self._wakeup_frame_count = 0

        # ── Legacy WAKEUP state ───────────────────────────────────
        self._wakeup_trigger_state = HandState.NONE
        self._wakeup_target_state = HandState.NONE
        self._consecutive_target = 0
        self._wakeup_frames: list = []

        # Legacy swipe tracking
        self._swipe_start_y = -1.0
        self._swipe_previous_y = -1.0
        self._swipe_consecutive_up = 0
        self._swipe_consecutive_down = 0
        self._swipe_cumulative = 0.0
        self._swipe_y_history: list = []

        # ── Cooldowns ─────────────────────────────────────────────
        self._last_gesture_time = 0.0
        self._last_swipe_time = 0.0

        # ── Stats ─────────────────────────────────────────────────
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
        if self._classifier:
            self._classifier.close()
            self._classifier = None
        logger.info("Gesture detector stopped")

    # ─────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────

    def _run(self):
        logger.info("Initializing camera and detectors...")

        # 1. MediaPipe (required for both modes)
        self._detector = HandLandmarkDetector()
        if not self._detector.is_initialized:
            logger.error("HandLandmarkDetector failed — aborting")
            return

        # 2. Decide detection mode
        user_wants_nn = config.DETECTION_METHOD == "neural_network"
        if user_wants_nn:
            try:
                self._classifier = GestureClassifier()
                if self._classifier.is_initialized:
                    self._active_mode = DetectionMode.NEURAL_NETWORK
                    logger.info("TCN model loaded — using Neural Network detection")
                else:
                    self._active_mode = DetectionMode.LEGACY
                    logger.warning(
                        "TCN model failed to initialize — "
                        "falling back to Legacy detection"
                    )
                    self._classifier = None
            except Exception as e:
                self._active_mode = DetectionMode.LEGACY
                logger.warning(
                    f"TCN model load error: {e} — "
                    f"falling back to Legacy detection"
                )
                self._classifier = None
        else:
            self._active_mode = DetectionMode.LEGACY
            logger.info("Using Legacy (rule-based) detection")

        # 3. Camera
        self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self._cap.isOpened():
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

        mode_label = self._active_mode.value.replace("_", " ").title()
        logger.info(f"Camera opened: {actual_w}x{actual_h} @ {fps:.0f}fps")
        logger.info(
            f"IDLE — scanning at ~{config.IDLE_FPS}fps ({mode_label})"
        )

        self._last_stats_time = time.time()
        last_frame_time = 0.0

        while self._running:
            now = time.time()

            min_interval = (
                config.IDLE_FRAME_INTERVAL_S
                if self._stage == Stage.IDLE
                else config.WAKEUP_FRAME_INTERVAL_S
            )
            elapsed_since_frame = now - last_frame_time
            if elapsed_since_frame < min_interval:
                time.sleep(max(0, min_interval - elapsed_since_frame - 0.002))
                continue

            ret, frame = self._cap.read()
            self._camera_frames += 1

            if not ret or frame is None:
                logger.warning("Camera read failed")
                time.sleep(0.1)
                continue

            last_frame_time = time.time()
            self._total_frames += 1

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                detail = self._detector.detect(frame_rgb)
            except Exception as e:
                logger.error(f"Detection error: {e}")
                continue

            try:
                if self._stage == Stage.IDLE:
                    self._process_idle(detail)
                else:
                    self._process_wakeup(detail)
            except Exception as e:
                logger.error(f"Processing error: {e}", exc_info=True)

            # Periodic stats
            if now - self._last_stats_time > 10:
                self._last_stats_time = now
                logger.info(
                    f"Stats: camera={self._camera_frames} "
                    f"processed={self._total_frames} "
                    f"stage={self._stage.value} "
                    f"mode={self._active_mode.value}"
                )

        if self._cap:
            self._cap.release()
        if self._detector:
            self._detector.close()
        if self._classifier:
            self._classifier.close()

    # ─────────────────────────────────────────────────────────────
    # IDLE
    # ─────────────────────────────────────────────────────────────

    def _process_idle(self, detail: DetectionDetail):
        self._idle_frame_count += 1

        if self._active_mode == DetectionMode.NEURAL_NETWORK:
            self._process_idle_tcn(detail)
        else:
            self._process_idle_legacy(detail)

    def _process_idle_tcn(self, detail: DetectionDetail):
        """TCN IDLE: wake up when any hand is seen consistently."""
        hand_detected = detail.hands_found > 0
        self._idle_window_hand.append(hand_detected)

        if self._idle_frame_count % 10 == 0:
            window_str = "".join("H" if h else "." for h in self._idle_window_hand)
            hand_count = sum(self._idle_window_hand)
            logger.info(
                f"IDLE #{self._idle_frame_count} | {window_str} | "
                f"hand={hand_count}/{config.IDLE_WINDOW_SIZE} | "
                f"{detail.summary()}"
            )

        if len(self._idle_window_hand) < config.IDLE_WINDOW_SIZE:
            return

        hand_count = sum(self._idle_window_hand)
        if hand_count >= config.IDLE_TRIGGER_THRESHOLD:
            self._enter_wakeup_tcn()

    def _process_idle_legacy(self, detail: DetectionDetail):
        """Legacy IDLE: detect palm or fist via rule-based classification."""
        self._idle_window_states.append(detail.state)

        if self._idle_frame_count % 10 == 0:
            window_str = "".join(
                self._state_emoji(s) for s in self._idle_window_states
            )
            palm_n = sum(1 for s in self._idle_window_states if s == HandState.PALM)
            fist_n = sum(1 for s in self._idle_window_states if s == HandState.FIST)
            logger.info(
                f"IDLE #{self._idle_frame_count} | {window_str} | "
                f"P={palm_n} F={fist_n} | {detail.summary()}"
            )

        if len(self._idle_window_states) < config.IDLE_WINDOW_SIZE:
            return

        palm_count = sum(
            1 for s in self._idle_window_states if s == HandState.PALM
        )
        fist_count = sum(
            1 for s in self._idle_window_states if s == HandState.FIST
        )

        if palm_count >= config.IDLE_TRIGGER_THRESHOLD:
            self._enter_wakeup_legacy(HandState.PALM, HandState.FIST)
        elif fist_count >= config.IDLE_TRIGGER_THRESHOLD:
            self._enter_wakeup_legacy(HandState.FIST, HandState.PALM)

    # ─────────────────────────────────────────────────────────────
    # WAKEUP — enter
    # ─────────────────────────────────────────────────────────────

    def _enter_wakeup_tcn(self):
        now = time.time()
        if now - self._last_gesture_time < config.GRAB_COOLDOWN_S:
            logger.info("Wakeup suppressed (cooldown)")
            return

        self._stage = Stage.WAKEUP
        self._wakeup_start_time = time.time()
        self._wakeup_frame_count = 0
        self._idle_window_hand.clear()
        self._idle_frame_count = 0

        if self._classifier:
            self._classifier.reset()

        logger.info(
            "WAKEUP! Hand detected — classifying with Neural Network..."
        )
        if self.on_stage_change:
            self.on_stage_change("WAKEUP", "🤚")

    def _enter_wakeup_legacy(self, trigger: HandState, target: HandState):
        now = time.time()
        if (now - self._last_gesture_time < config.GRAB_COOLDOWN_S
                and now - self._last_swipe_time < config.SWIPE_COOLDOWN_S):
            logger.info("Wakeup suppressed (cooldown)")
            return

        self._stage = Stage.WAKEUP
        self._wakeup_start_time = time.time()
        self._wakeup_trigger_state = trigger
        self._wakeup_target_state = target
        self._consecutive_target = 0
        self._wakeup_frames = []
        self._wakeup_frame_count = 0
        self._idle_window_states.clear()
        self._idle_frame_count = 0

        # Reset swipe state
        self._swipe_start_y = -1.0
        self._swipe_previous_y = -1.0
        self._swipe_consecutive_up = 0
        self._swipe_consecutive_down = 0
        self._swipe_cumulative = 0.0
        self._swipe_y_history = []

        trigger_e = "🖐" if trigger == HandState.PALM else "✊"
        target_e = "✊" if target == HandState.FIST else "🖐"
        motion = "GRAB" if target == HandState.FIST else "RELEASE"

        logger.info(
            f"WAKEUP! Detected {trigger_e} — "
            f"watching {config.WAKEUP_DURATION_S}s for "
            f"{trigger_e} -> {target_e} ({motion}) or swipe"
        )

        indicator = "✊" if target == HandState.FIST else "🤚"
        if self.on_stage_change:
            self.on_stage_change("WAKEUP", indicator)

    # ─────────────────────────────────────────────────────────────
    # WAKEUP — process
    # ─────────────────────────────────────────────────────────────

    def _process_wakeup(self, detail: DetectionDetail):
        self._wakeup_frame_count += 1

        if self._active_mode == DetectionMode.NEURAL_NETWORK:
            self._process_wakeup_tcn(detail)
        else:
            self._process_wakeup_legacy(detail)

    # ── TCN wakeup ────────────────────────────────────────────────

    def _process_wakeup_tcn(self, detail: DetectionDetail):
        elapsed = time.time() - self._wakeup_start_time

        if (detail.hands_found > 0
                and detail.raw_landmarks is not None
                and self._classifier is not None
                and self._classifier.is_initialized):

            result = self._classifier.add_frame_and_classify(
                detail.raw_landmarks
            )

            if (result is not None
                    and result.confidence >= config.TCN_CONFIDENCE_THRESHOLD
                    and result.gesture != "noise"):

                event = {
                    "grab": GestureEvent.GRAB,
                    "release": GestureEvent.RELEASE,
                    "swipe_up": GestureEvent.SWIPE_UP,
                    "swipe_down": GestureEvent.SWIPE_DOWN,
                }.get(result.gesture)

                if event is not None:
                    logger.info(
                        f"{result.gesture.upper()} detected! "
                        f"conf={result.confidence:.2f} "
                        f"frames={result.valid_frames}"
                    )
                    self._last_gesture_time = time.time()
                    self._exit_wakeup()
                    self._fire_gesture(event)
                    return

            # Progress logging
            if self._wakeup_frame_count % 5 == 0:
                remaining = config.WAKEUP_DURATION_S - elapsed
                if result is not None:
                    info = (
                        f"{result.gesture}({result.confidence:.2f}) "
                        f"v={result.valid_frames}"
                    )
                else:
                    info = "collecting..."
                logger.info(
                    f"WK #{self._wakeup_frame_count} "
                    f"{remaining:.1f}s | {info} | {detail.summary()}"
                )
        else:
            if self._wakeup_frame_count % 5 == 0:
                remaining = config.WAKEUP_DURATION_S - elapsed
                logger.info(
                    f"WK #{self._wakeup_frame_count} "
                    f"{remaining:.1f}s | no hand | {detail.summary()}"
                )

        if elapsed > config.WAKEUP_DURATION_S:
            logger.info("WAKEUP timeout")
            self._exit_wakeup()

    # ── Legacy wakeup ─────────────────────────────────────────────

    def _process_wakeup_legacy(self, detail: DetectionDetail):
        self._wakeup_frames.append(detail.state)
        elapsed = time.time() - self._wakeup_start_time
        remaining = config.WAKEUP_DURATION_S - elapsed

        # 1. Check swipe
        swipe_event = None
        if detail.hands_found > 0 and self._wakeup_frame_count > 3:
            swipe_event = self._check_swipe_legacy(detail)

        if swipe_event is not None:
            arrow = "UP" if swipe_event == GestureEvent.SWIPE_UP else "DOWN"
            logger.info(
                f"SWIPE {arrow} CONFIRMED! "
                f"(disp={self._swipe_cumulative:.3f})"
            )
            self._last_swipe_time = time.time()
            self._last_gesture_time = time.time()
            self._exit_wakeup()
            self._fire_gesture(swipe_event)
            return

        # 2. Check grab/release
        if detail.state == self._wakeup_target_state:
            self._consecutive_target += 1
        else:
            self._consecutive_target = 0

        # Logging
        if (self._wakeup_frame_count % 5 == 0
                or self._consecutive_target > 0):
            state_e = self._state_emoji(detail.state)
            progress = self._progress_bar(
                self._consecutive_target, config.WAKEUP_CONFIRM_FRAMES
            )
            swipe_info = (
                f"dy={self._swipe_cumulative:.3f} "
                f"up={self._swipe_consecutive_up} "
                f"dn={self._swipe_consecutive_down}"
            )
            logger.info(
                f"WK #{self._wakeup_frame_count} {remaining:.1f}s | "
                f"{state_e} str={self._consecutive_target}/"
                f"{config.WAKEUP_CONFIRM_FRAMES} {progress} | "
                f"{swipe_info} | {detail.summary()}"
            )

        # 3. Timeout
        if elapsed > config.WAKEUP_DURATION_S:
            summary = self._wakeup_summary()
            logger.info(f"WAKEUP timeout — {summary}")
            self._exit_wakeup()
            return

        # 4. Confirm
        if self._consecutive_target >= config.WAKEUP_CONFIRM_FRAMES:
            if self._wakeup_target_state == HandState.FIST:
                logger.info("GRAB CONFIRMED! (palm -> fist)")
                event = GestureEvent.GRAB
            else:
                logger.info("RELEASE CONFIRMED! (fist -> palm)")
                event = GestureEvent.RELEASE

            self._last_gesture_time = time.time()
            self._exit_wakeup()
            self._fire_gesture(event)

    # ── Legacy swipe detection ────────────────────────────────────

    def _check_swipe_legacy(
        self, detail: DetectionDetail
    ) -> Optional[GestureEvent]:
        now = time.time()
        if now - self._last_swipe_time < config.SWIPE_COOLDOWN_S:
            return None

        current_y = detail.center_y
        self._swipe_y_history.append(current_y)

        if self._swipe_start_y < 0:
            self._swipe_start_y = current_y
            self._swipe_previous_y = current_y
            return None

        frame_velocity = current_y - self._swipe_previous_y
        self._swipe_previous_y = current_y
        self._swipe_cumulative = current_y - self._swipe_start_y

        if frame_velocity < -config.SWIPE_MIN_VELOCITY:
            self._swipe_consecutive_up += 1
            self._swipe_consecutive_down = 0
        elif frame_velocity > config.SWIPE_MIN_VELOCITY:
            self._swipe_consecutive_down += 1
            self._swipe_consecutive_up = 0

        total_disp = abs(self._swipe_cumulative)
        has_enough_frames = (
            self._swipe_consecutive_up >= config.SWIPE_CONFIRM_FRAMES
            or self._swipe_consecutive_down >= config.SWIPE_CONFIRM_FRAMES
        )
        has_enough_displacement = (
            total_disp >= config.SWIPE_DISPLACEMENT_THRESHOLD
        )

        if has_enough_frames and has_enough_displacement:
            if self._swipe_cumulative < 0:
                return GestureEvent.SWIPE_UP
            else:
                return GestureEvent.SWIPE_DOWN

        # Trend analysis
        min_trend_frames = config.SWIPE_CONFIRM_FRAMES + 3
        if len(self._swipe_y_history) >= min_trend_frames:
            recent = self._swipe_y_history[-min_trend_frames:]
            start_avg = sum(recent[:3]) / 3.0
            end_avg = sum(recent[-3:]) / 3.0
            trend = end_avg - start_avg

            if abs(trend) >= config.SWIPE_DISPLACEMENT_THRESHOLD * 1.2:
                if trend < 0:
                    return GestureEvent.SWIPE_UP
                else:
                    return GestureEvent.SWIPE_DOWN

        return None

    # ─────────────────────────────────────────────────────────────
    # Exit wakeup
    # ─────────────────────────────────────────────────────────────

    def _exit_wakeup(self):
        self._stage = Stage.IDLE
        self._wakeup_frame_count = 0
        self._idle_window_states.clear()
        self._idle_window_hand.clear()
        self._idle_frame_count = 0

        # Legacy state
        self._consecutive_target = 0
        self._wakeup_frames = []
        self._swipe_start_y = -1.0
        self._swipe_previous_y = -1.0
        self._swipe_consecutive_up = 0
        self._swipe_consecutive_down = 0
        self._swipe_cumulative = 0.0
        self._swipe_y_history = []

        # TCN state
        if self._classifier:
            self._classifier.reset()

        if self.on_stage_change:
            self.on_stage_change("IDLE", "")

        logger.info("Back to IDLE")

    # ─────────────────────────────────────────────────────────────
    # Fire gesture
    # ─────────────────────────────────────────────────────────────

    def _fire_gesture(self, event: GestureEvent):
        """Fire gesture event in a separate thread to not block camera."""
        if self.on_gesture:
            threading.Thread(
                target=self.on_gesture,
                args=(event,),
                daemon=True,
            ).start()

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _state_emoji(state: HandState) -> str:
        return {
            HandState.PALM: "P",
            HandState.FIST: "F",
            HandState.UNKNOWN: "?",
            HandState.NONE: ".",
        }.get(state, "?")

    @staticmethod
    def _progress_bar(current: int, maximum: int) -> str:
        filled = min(current, maximum)
        empty = maximum - filled
        return "[" + "#" * filled + "-" * empty + "]"

    def _wakeup_summary(self) -> str:
        total = len(self._wakeup_frames)
        palm = sum(1 for s in self._wakeup_frames if s == HandState.PALM)
        fist = sum(1 for s in self._wakeup_frames if s == HandState.FIST)
        timeline = "".join(
            self._state_emoji(s) for s in self._wakeup_frames[-30:]
        )
        return (
            f"total={total} P={palm} F={fist} "
            f"dy={self._swipe_cumulative:.3f} | {timeline}"
        )
