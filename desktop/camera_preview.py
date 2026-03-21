#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*
#*   File:         camera_preview.py
#*   Author:       XUranus
#*   Date:         2026-03-20
#*   Description:  Camera preview window with MediaPipe hand skeleton
#*
#================================================================*/

import logging
import threading
import time
import math
from typing import Optional, List, Tuple

import cv2
import numpy as np
import mediapipe as mp

import config
from hand_landmark import HandLandmarkDetector, HandState

logger = logging.getLogger("CameraPreview")


# MediaPipe hand connections for drawing skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),            # Palm
]

# Finger tip and pip indices for coloring
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

# Finger names for display
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


class CameraPreview:
    """Camera preview window showing hand skeleton detection."""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._hands = None
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_hands = mp.solutions.hands

        # Hand landmark detector for classification
        self._detector: Optional[HandLandmarkDetector] = None

        # Frame dimensions
        self._width = config.CAMERA_WIDTH
        self._height = config.CAMERA_HEIGHT

        # Window name
        self._window_name = "AirGesture - Hand Detection"

        # FPS tracking
        self._last_frame_time = 0.0
        self._fps = 0.0

        # Frame callback for gesture detector
        self.on_frame: Optional[callable] = None  # callback(frame_rgb, timestamp)

    def start(self):
        """Start the camera preview in a separate thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Camera preview started")

    def stop(self):
        """Stop the camera preview."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass
        logger.info("Camera preview stopped")

    def _run(self):
        """Main loop for camera preview."""
        # Initialize MediaPipe Hands
        try:
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0,
            )
            logger.info("MediaPipe Hands initialized for preview")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            return

        # Initialize hand landmark detector for classification
        self._detector = HandLandmarkDetector()
        if not self._detector.is_initialized:
            logger.error("HandLandmarkDetector failed for preview")
            return

        # Open camera
        self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self._cap.isOpened():
            for idx in range(4):
                if idx == config.CAMERA_INDEX:
                    continue
                self._cap = cv2.VideoCapture(idx)
                if self._cap.isOpened():
                    logger.info(f"Camera {idx} opened for preview")
                    break
            else:
                logger.error("No camera found for preview")
                return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera preview: {actual_w}x{actual_h}")

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, 640, 480)

        self._last_frame_time = time.time()

        while self._running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            # Calculate FPS
            now = time.time()
            dt = now - self._last_frame_time
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            self._last_frame_time = now

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            try:
                results = self._hands.process(frame_rgb)
            except Exception as e:
                logger.error(f"MediaPipe process error: {e}")
                continue

            # Get classification using our detector
            detail = self._detector.detect(frame_rgb)

            # Draw landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self._draw_hand_skeleton(frame, hand_landmarks)

            # Draw status overlay
            self._draw_status_overlay(frame, detail, results)

            # Call frame callback for gesture detector
            if self.on_frame:
                try:
                    self.on_frame(frame_rgb, now)
                except Exception as e:
                    logger.error(f"Frame callback error: {e}")

            # Show frame
            cv2.imshow(self._window_name, frame)

            # Check for 'q' key to close
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                logger.info("Preview window closed by user")
                break

        # Cleanup
        if self._cap:
            self._cap.release()
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass
        if self._hands:
            self._hands.close()
        if self._detector:
            self._detector.close()

    def _draw_hand_skeleton(self, frame: np.ndarray, hand_landmarks):
        """Draw hand skeleton with custom styling."""
        h, w, _ = frame.shape

        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = hand_landmarks.landmark[start_idx]
            end = hand_landmarks.landmark[end_idx]

            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))

            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # Draw landmarks
        for i, landmark in enumerate(hand_landmarks.landmark):
            x, y = int(landmark.x * w), int(landmark.y * h)

            # Different colors for different parts
            if i in FINGER_TIPS:
                # Finger tips - red
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
                cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
            elif i in FINGER_PIPS:
                # PIP joints - yellow
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
            elif i == 0:
                # Wrist - blue
                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)
            else:
                # Other landmarks - green
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw palm center
        palm_indices = [0, 5, 9, 13, 17]
        cx = sum(hand_landmarks.landmark[i].x for i in palm_indices) / len(palm_indices)
        cy = sum(hand_landmarks.landmark[i].y for i in palm_indices) / len(palm_indices)
        cv2.circle(frame, (int(cx * w), int(cy * h)), 12, (255, 0, 255), 2)

    def _draw_status_overlay(self, frame: np.ndarray, detail, results):
        """Draw status information overlay on the frame."""
        h, w, _ = frame.shape

        # Create semi-transparent background for status panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw FPS
        cv2.putText(
            frame, f"FPS: {self._fps:.1f}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

        # Draw hand state with color and emoji
        state_colors = {
            HandState.PALM: (0, 255, 0),     # Green
            HandState.FIST: (0, 165, 255),   # Orange
            HandState.UNKNOWN: (0, 255, 255), # Yellow
            HandState.NONE: (128, 128, 128),  # Gray
        }

        state_emojis = {
            HandState.PALM: "PALM",
            HandState.FIST: "FIST",
            HandState.UNKNOWN: "???",
            HandState.NONE: "NO HAND",
        }

        state_color = state_colors.get(detail.state, (255, 255, 255))
        state_text = state_emojis.get(detail.state, detail.state.value)

        # Draw state with background
        cv2.putText(
            frame, f"State: {state_text}",
            (w - 180, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, state_color, 2
        )

        # Draw confidence if hand detected
        if detail.hands_found > 0:
            cv2.putText(
                frame, f"Conf: {detail.confidence:.2f}",
                (w - 180, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1
            )

        # Draw extended/curled finger count
        if detail.hands_found > 0:
            ext_text = f"Extended: {detail.extended_count}  Curled: {detail.curled_count}"
            cv2.putText(
                frame, ext_text,
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1
            )

        # Draw action suggestion
        action_text, action_color = self._get_action_suggestion(detail)
        if action_text:
            cv2.putText(
                frame, action_text,
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, action_color, 2
            )

        # Draw bottom instruction bar
        cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(
            frame, "Press 'q' to close | PALM->FIST = GRAB | FIST->PALM = RELEASE",
            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )

    def _get_action_suggestion(self, detail) -> Tuple[str, Tuple[int, int, int]]:
        """Get action suggestion based on hand state."""
        if detail.state == HandState.NONE:
            return "Show your hand to start...", (150, 150, 150)
        elif detail.state == HandState.PALM:
            if detail.extended_count >= 4:
                return "Ready for GRAB (make fist)", (0, 255, 0)
            return "Open hand detected", (0, 200, 100)
        elif detail.state == HandState.FIST:
            return "Ready for RELEASE (open hand)", (0, 165, 255)
        elif detail.state == HandState.UNKNOWN:
            if detail.extended_count > detail.curled_count:
                return "Partially open...", (0, 255, 255)
            else:
                return "Partially closed...", (100, 200, 200)
        return "", (255, 255, 255)
