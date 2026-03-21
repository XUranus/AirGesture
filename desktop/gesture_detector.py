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

# Try to import TCN classifier
try:
    from tcn_classifier import TCNClassifier
    TCN_AVAILABLE = True
except ImportError:
    TCN_AVAILABLE = False
    logger.warning("TCNClassifier not available, using rule-based detection")


class GestureEvent(Enum):
    GRAB = "GRAB"
    RELEASE = "RELEASE"
    SWIPE_UP = "SWIPE_UP"
    SWIPE_DOWN = "SWIPE_DOWN"


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
        self._use_shared_camera = False  # Whether using shared camera from CameraPreview

        # TCN Classifier (optional)
        self._tcn: Optional[TCNClassifier] = None
        self._use_tcn = False
        if config.USE_TCN_CLASSIFIER and TCN_AVAILABLE:
            try:
                self._tcn = TCNClassifier(
                    config_path=config.TCN_MODEL_CONFIG,
                    model_path=config.TCN_MODEL_PATH,
                    threshold=config.TCN_THRESHOLD,
                    smooth_window=config.TCN_SMOOTH_WINDOW,
                    window_seconds=config.TCN_WINDOW_SECONDS,
                    fill_frames=config.TCN_FILL_FRAMES,
                )
                if self._tcn.is_available:
                    self._use_tcn = True
                    logger.info(f"🤖 TCN classifier enabled: {self._tcn.class_names}")
                else:
                    logger.info("⚠️ TCN classifier not available, using rule-based detection")
            except Exception as e:
                logger.warning(f"Failed to initialize TCN classifier: {e}")

        # Stage
        self._stage = Stage.IDLE

        # Idle state
        self._idle_window: deque = deque(maxlen=config.IDLE_WINDOW_SIZE)
        self._idle_frame_count = 0

        # Wakeup state — grab/release
        self._wakeup_start_time = 0.0
        self._wakeup_trigger_state = HandState.NONE
        self._wakeup_target_state = HandState.NONE
        self._consecutive_target = 0
        self._wakeup_frames: list = []
        self._wakeup_frame_count = 0

        # Wakeup state — swipe tracking
        self._swipe_start_y = -1.0
        self._swipe_previous_y = -1.0
        self._swipe_consecutive_up = 0
        self._swipe_consecutive_down = 0
        self._swipe_cumulative = 0.0
        self._swipe_y_history: list = []

        # Cooldowns
        self._last_gesture_time = 0.0
        self._last_swipe_time = 0.0

        # Stats
        self._total_frames = 0
        self._camera_frames = 0
        self._last_stats_time = 0.0

    def start(self, use_shared_camera: bool = False):
        """Start gesture detection.

        Args:
            use_shared_camera: If True, use frames from CameraPreview callback
        """
        if self._running:
            return

        self._running = True
        self._use_shared_camera = use_shared_camera

        # Initialize detector (needed for both modes)
        self._detector = HandLandmarkDetector()
        if not self._detector.is_initialized:
            logger.error("HandLandmarkDetector failed — aborting")
            self._running = False
            return

        if use_shared_camera:
            # Will receive frames via process_frame() callback
            logger.info("🤖 Using shared camera from CameraPreview")
            logger.info(f"👁️ IDLE stage — waiting for frames")
        else:
            # Start own camera thread
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def process_frame(self, frame_rgb: np.ndarray, timestamp: float):
        """Process a frame from shared camera source.

        Called by CameraPreview when on_frame is set.
        """
        if not self._running or not self._use_shared_camera:
            return

        self._total_frames += 1
        current_t = timestamp

        # Get detection detail and landmarks
        detail, landmarks = self._detector.detect_with_landmarks(frame_rgb)

        # TCN-based detection
        tcn_gesture = None
        tcn_conf = 0.0

        if self._use_tcn and self._tcn is not None:
            if landmarks is not None:
                lm_array = np.array(landmarks, dtype=np.float32).flatten()
                self._tcn.push(lm_array, current_t)
            else:
                self._tcn.push_missing(current_t)

            # Get TCN prediction
            tcn_gesture, tcn_conf = self._tcn.predict(current_t)

            # Process TCN gesture
            if tcn_gesture is not None and tcn_gesture != "unknown":
                event_name = self._tcn.get_gesture_event(tcn_gesture)
                if event_name:
                    is_swipe = event_name in ("SWIPE_UP", "SWIPE_DOWN")
                    cooldown = config.SWIPE_COOLDOWN_S if is_swipe else config.GRAB_COOLDOWN_S
                    if current_t - self._last_gesture_time >= cooldown:
                        logger.info(f"🤖 TCN: {tcn_gesture.upper()} ({tcn_conf:.2f})")
                        self._last_gesture_time = current_t
                        self._fire_gesture(GestureEvent[event_name])
                        return

        # Fallback to rule-based detection
        try:
            if self._stage == Stage.IDLE:
                self._process_idle(detail)
            else:
                self._process_wakeup(detail)
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)

        # Periodic stats
        if current_t - self._last_stats_time > 10:
            self._last_stats_time = current_t
            logger.info(f"📊 Stats: processed={self._total_frames} stage={self._stage.value}")

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
        if self._tcn:
            self._tcn.reset()
        logger.info("Gesture detector stopped")

    def _run(self):
        logger.info("Initializing camera and MediaPipe...")

        self._detector = HandLandmarkDetector()
        if not self._detector.is_initialized:
            logger.error("HandLandmarkDetector failed — aborting")
            return

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

        logger.info(f"📷 Camera opened: {actual_w}x{actual_h} @ {fps:.0f}fps")

        if self._use_tcn:
            logger.info(f"🤖 Using TCN classifier (threshold={config.TCN_THRESHOLD}, window={config.TCN_WINDOW_SECONDS}s)")
        else:
            logger.info(f"👁️ Using rule-based detection")

        logger.info(f"👁️ IDLE stage — scanning at ~{config.IDLE_FPS}fps (swipe+grab+release)")

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

            # Get hand detection and landmarks in one call
            current_t = time.time()
            tcn_gesture = None
            tcn_conf = 0.0
            detail = None
            landmarks = None

            try:
                if self._use_tcn and self._tcn is not None:
                    # Use detect_with_landmarks to get both in one call
                    detail, landmarks = self._detector.detect_with_landmarks(frame_rgb)
                else:
                    detail = self._detector.detect(frame_rgb)
            except Exception as e:
                logger.error(f"Detection error: {e}")
                continue

            # TCN-based detection
            if self._use_tcn and self._tcn is not None:
                if landmarks is not None:
                    lm_array = np.array(landmarks, dtype=np.float32).flatten()
                    self._tcn.push(lm_array, current_t)
                else:
                    self._tcn.push_missing(current_t)

                # Get TCN prediction
                tcn_gesture, tcn_conf = self._tcn.predict(current_t)

                # Periodic status log (every 30 frames when collecting)
                if self._total_frames % 30 == 0:
                    buf_len = len(self._tcn.buffer) if self._tcn.buffer else 0
                    buf_cov = self._tcn.buffer.coverage_seconds(current_t) if self._tcn.buffer else 0
                    logger.debug(f"TCN buffer: {buf_len} frames, {buf_cov:.2f}s coverage")

                # Process TCN gesture
                if tcn_gesture is not None and tcn_gesture != "unknown":
                    event_name = self._tcn.get_gesture_event(tcn_gesture)
                    if event_name:
                        # Check cooldown based on gesture type
                        is_swipe = event_name in ("SWIPE_UP", "SWIPE_DOWN")
                        cooldown = config.SWIPE_COOLDOWN_S if is_swipe else config.GRAB_COOLDOWN_S
                        if current_t - self._last_gesture_time >= cooldown:
                            logger.info(f"🤖 TCN: {tcn_gesture.upper()} ({tcn_conf:.2f})")
                            self._last_gesture_time = current_t
                            self._fire_gesture(GestureEvent[event_name])
                            continue

            # Fallback to rule-based detection
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
                    f"📊 Stats: camera={self._camera_frames} "
                    f"processed={self._total_frames} "
                    f"stage={self._stage.value}"
                )

        if self._cap:
            self._cap.release()
        if self._detector:
            self._detector.close()

    # ─── IDLE ───

    def _process_idle(self, detail: DetectionDetail):
        self._idle_frame_count += 1
        self._idle_window.append(detail.state)

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

        palm_count = sum(1 for s in self._idle_window if s == HandState.PALM)
        fist_count = sum(1 for s in self._idle_window if s == HandState.FIST)

        if palm_count >= config.IDLE_TRIGGER_THRESHOLD:
            self._enter_wakeup(HandState.PALM, HandState.FIST)
        elif fist_count >= config.IDLE_TRIGGER_THRESHOLD:
            self._enter_wakeup(HandState.FIST, HandState.PALM)

    # ─── ENTER WAKEUP ───

    def _enter_wakeup(self, trigger: HandState, target: HandState):
        now = time.time()
        if now - self._last_gesture_time < config.GRAB_COOLDOWN_S:
            logger.info("⏳ Wakeup suppressed — cooldown")
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
            f"🔔 WAKEUP! Detected {trigger_e} — "
            f"watching 2s for {trigger_e}→{target_e} ({motion}) or ↑↓ swipe"
        )

        indicator = "✊" if target == HandState.FIST else "🤚"
        if self.on_stage_change:
            self.on_stage_change("WAKEUP", indicator)

    # ─── PROCESS WAKEUP ───

    def _process_wakeup(self, detail: DetectionDetail):
        self._wakeup_frame_count += 1
        self._wakeup_frames.append(detail.state)

        elapsed = time.time() - self._wakeup_start_time
        remaining = config.WAKEUP_DURATION_S - elapsed

        # ─── 1. Check SWIPE (only after a few warmup frames) ───
        swipe_event = None
        if detail.hands_found > 0 and self._wakeup_frame_count > 3:
            swipe_event = self._check_swipe(detail)

        if swipe_event is not None:
            arrow = "⬆️" if swipe_event == GestureEvent.SWIPE_UP else "⬇️"
            logger.info(
                f"✅ {arrow} SWIPE "
                f"{'UP' if swipe_event == GestureEvent.SWIPE_UP else 'DOWN'} "
                f"CONFIRMED! (disp={self._swipe_cumulative:.3f})"
            )
            self._last_swipe_time = time.time()
            self._last_gesture_time = time.time()
            self._exit_wakeup()
            self._fire_gesture(swipe_event)
            return

        # ─── 2. Check GRAB/RELEASE ───
        if detail.state == self._wakeup_target_state:
            self._consecutive_target += 1
        else:
            if self._consecutive_target > 0:
                logger.debug(
                    f"Streak broken at {self._consecutive_target} "
                    f"(got {detail.state.value}, need {self._wakeup_target_state.value})"
                )
            self._consecutive_target = 0

        # ─── Logging ───
        if (self._wakeup_frame_count % 5 == 0
                or self._consecutive_target > 0
                or swipe_event is not None):
            state_e = self._state_emoji(detail.state)
            progress = self._progress_bar(
                self._consecutive_target, config.WAKEUP_CONFIRM_FRAMES
            )
            swipe_info = (
                f"dy={self._swipe_cumulative:.3f} "
                f"↑={self._swipe_consecutive_up} "
                f"↓={self._swipe_consecutive_down}"
            )
            logger.info(
                f"⏱️ WK #{self._wakeup_frame_count} "
                f"{remaining:.1f}s | {state_e} "
                f"str={self._consecutive_target}/"
                f"{config.WAKEUP_CONFIRM_FRAMES} "
                f"{progress} | {swipe_info} | "
                f"{detail.summary()}"
            )

        # ─── 3. Timeout ───
        if elapsed > config.WAKEUP_DURATION_S:
            summary = self._wakeup_summary()
            logger.info(f"⌛ WAKEUP timeout — no gesture. {summary}")
            self._exit_wakeup()
            return

        # ─── 4. Confirm GRAB/RELEASE ───
        if self._consecutive_target >= config.WAKEUP_CONFIRM_FRAMES:
            if self._wakeup_target_state == HandState.FIST:
                logger.info("✅ ✊ GRAB CONFIRMED! (palm→fist)")
                event = GestureEvent.GRAB
            else:
                logger.info("✅ 🖐 RELEASE CONFIRMED! (fist→palm)")
                event = GestureEvent.RELEASE

            self._last_gesture_time = time.time()
            self._exit_wakeup()
            self._fire_gesture(event)

    # ─── SWIPE DETECTION ───

    def _check_swipe(self, detail: DetectionDetail) -> Optional[GestureEvent]:
        # Cooldown
        now = time.time()
        if now - self._last_swipe_time < config.SWIPE_COOLDOWN_S:
            return None

        # Get hand center Y position
        try:
            current_y = detail.center_y
        except AttributeError:
            # Fallback if DetectionDetail doesn't have center_y
            logger.debug("DetectionDetail missing center_y")
            return None

        self._swipe_y_history.append(current_y)

        # Initialize start position
        if self._swipe_start_y < 0:
            self._swipe_start_y = current_y
            self._swipe_previous_y = current_y
            return None

        # Frame-to-frame velocity
        # In image coords: Y=0 is top, Y=1 is bottom
        # frame_velocity > 0 means hand moving DOWN in image
        # frame_velocity < 0 means hand moving UP in image
        frame_velocity = current_y - self._swipe_previous_y
        self._swipe_previous_y = current_y

        # Cumulative displacement from start
        self._swipe_cumulative = current_y - self._swipe_start_y

        # Count consecutive directional frames
        if frame_velocity < -config.SWIPE_MIN_VELOCITY:
            # Moving UP in image
            self._swipe_consecutive_up += 1
            self._swipe_consecutive_down = 0
        elif frame_velocity > config.SWIPE_MIN_VELOCITY:
            # Moving DOWN in image
            self._swipe_consecutive_down += 1
            self._swipe_consecutive_up = 0
        else:
            # Not enough movement this frame — don't reset streaks,
            # allow brief pauses in motion.
            # But if both streaks are 0, this means no direction established yet.
            pass

        total_disp = abs(self._swipe_cumulative)

        # ─── Method 1: Consecutive frames + displacement ───
        has_enough_frames = (
            self._swipe_consecutive_up >= config.SWIPE_CONFIRM_FRAMES
            or self._swipe_consecutive_down >= config.SWIPE_CONFIRM_FRAMES
        )
        has_enough_displacement = total_disp >= config.SWIPE_DISPLACEMENT_THRESHOLD

        if has_enough_frames and has_enough_displacement:
            if self._swipe_cumulative < 0:
                return GestureEvent.SWIPE_UP
            else:
                return GestureEvent.SWIPE_DOWN

        # ─── Method 2: Trend analysis over recent frames ───
        min_trend_frames = config.SWIPE_CONFIRM_FRAMES + 3  # need a bit more for trend
        if len(self._swipe_y_history) >= min_trend_frames:
            recent = self._swipe_y_history[-min_trend_frames:]
            start_avg = sum(recent[:3]) / 3.0
            end_avg = sum(recent[-3:]) / 3.0
            trend = end_avg - start_avg

            # Require larger displacement for trend method (less reliable)
            if abs(trend) >= config.SWIPE_DISPLACEMENT_THRESHOLD * 1.2:
                if trend < 0:
                    return GestureEvent.SWIPE_UP
                else:
                    return GestureEvent.SWIPE_DOWN

        return None

    # ─── EXIT WAKEUP ───

    def _exit_wakeup(self):
        self._stage = Stage.IDLE
        self._consecutive_target = 0
        self._wakeup_frames = []
        self._wakeup_frame_count = 0
        self._idle_window.clear()
        self._idle_frame_count = 0

        # Reset swipe state
        self._swipe_start_y = -1.0
        self._swipe_previous_y = -1.0
        self._swipe_consecutive_up = 0
        self._swipe_consecutive_down = 0
        self._swipe_cumulative = 0.0
        self._swipe_y_history = []

        if self.on_stage_change:
            self.on_stage_change("IDLE", "")

        logger.info("👁️ Back to IDLE — scanning at ~10fps")

    # ─── FIRE GESTURE ───

    def _fire_gesture(self, event: GestureEvent):
        """Fire gesture event in a separate thread to not block camera."""
        if self.on_gesture:
            threading.Thread(
                target=self.on_gesture,
                args=(event,),
                daemon=True,
            ).start()

    # ─── HELPERS ───

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
            self._state_emoji(s)
            for s in self._wakeup_frames[-30:]
        )
        return (
            f"total={total} 🖐={palm} ✊={fist} ·={none} ❓={unk} "
            f"swipe_dy={self._swipe_cumulative:.3f} | {timeline}"
        )
