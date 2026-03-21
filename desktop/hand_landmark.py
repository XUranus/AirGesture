#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         hand_landmark.py
#*   Author:       XUranus
#*   Date:         2026-03-14
#*   Description:  
#*
#================================================================*/
# /GrabDrop-Desktop/hand_landmark.py
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List

import mediapipe as mp
import numpy as np

import config

logger = logging.getLogger("HandLandmark")


class HandState(Enum):
    NONE = "NONE"
    PALM = "PALM"
    FIST = "FIST"
    UNKNOWN = "UNKNOWN"


FINGER_NAMES = ["IDX", "MID", "RNG", "PNK"]


@dataclass
class DetectionDetail:
    state: HandState
    finger_ratios: List[float] = field(default_factory=list)
    extended_count: int = 0
    curled_count: int = 0
    hands_found: int = 0
    confidence: float = 0.0
    handedness: str = "?"
    center_x: float = 0.5
    center_y: float = 0.5
    wrist_x: float = 0.5
    wrist_y: float = 0.5

    def summary(self) -> str:
        if self.hands_found == 0:
            return "NO_HAND"
        ratios = " ".join(
            f"{n}:{r:.2f}"
            for n, r in zip(FINGER_NAMES, self.finger_ratios)
        )
        return (
            f"{self.state.value} e={self.extended_count} "
            f"c={self.curled_count} conf={self.confidence:.2f} "
            f"pos=({self.center_x:.2f},{self.center_y:.2f}) "
            f"[{ratios}]"
        )


class HandLandmarkDetector:
    WRIST = 0
    INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
    RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
    PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

    FINGER_LANDMARKS = [
        (INDEX_TIP, INDEX_PIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        (RING_TIP, RING_PIP, RING_MCP),
        (PINKY_TIP, PINKY_PIP, PINKY_MCP),
    ]

    CENTER_INDICES = [0, 5, 9, 13, 17]  # WRIST + 4 MCPs

    def __init__(self):
        self.hands = None
        self.is_initialized = False
        try:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
                model_complexity=0,
            )
            self.is_initialized = True
            logger.info("MediaPipe Hands initialized (lite model)")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")

    def detect(self, frame_rgb: np.ndarray) -> DetectionDetail:
        if not self.is_initialized or self.hands is None:
            return DetectionDetail(state=HandState.NONE)

        try:
            results = self.hands.process(frame_rgb)
            return self._classify(results)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return DetectionDetail(state=HandState.NONE)

    def detect_with_landmarks(self, frame_rgb: np.ndarray) -> tuple:
        """Detect hand and return both classification and raw landmarks for drawing.

        Returns:
            tuple: (DetectionDetail, landmarks_list or None)
            landmarks_list is a list of (x, y) tuples in normalized coordinates
        """
        if not self.is_initialized or self.hands is None:
            return DetectionDetail(state=HandState.NONE), None

        try:
            results = self.hands.process(frame_rgb)
            detail = self._classify(results)

            landmarks = None
            if results.multi_hand_landmarks:
                raw_landmarks = results.multi_hand_landmarks[0].landmark
                landmarks = [(lm.x, lm.y, lm.z) for lm in raw_landmarks]

            return detail, landmarks
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return DetectionDetail(state=HandState.NONE), None

    def _classify(self, results) -> DetectionDetail:
        if not results.multi_hand_landmarks:
            return DetectionDetail(state=HandState.NONE, hands_found=0)

        landmarks = results.multi_hand_landmarks[0].landmark
        if len(landmarks) < 21:
            return DetectionDetail(
                state=HandState.UNKNOWN,
                hands_found=len(results.multi_hand_landmarks),
            )

        confidence = 0.0
        handedness = "?"
        if results.multi_handedness:
            h = results.multi_handedness[0].classification[0]
            confidence = h.score
            handedness = h.label

        wrist = landmarks[self.WRIST]

        # Calculate hand center
        cx = sum(landmarks[i].x for i in self.CENTER_INDICES) / len(self.CENTER_INDICES)
        cy = sum(landmarks[i].y for i in self.CENTER_INDICES) / len(self.CENTER_INDICES)

        extended = 0
        curled = 0
        ratios = []

        for tip_idx, pip_idx, mcp_idx in self.FINGER_LANDMARKS:
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]

            tip_to_wrist = self._dist(tip.x, tip.y, wrist.x, wrist.y)
            mcp_to_wrist = self._dist(mcp.x, mcp.y, wrist.x, wrist.y)

            if mcp_to_wrist < 0.001:
                ratios.append(0.0)
                continue

            ratio = tip_to_wrist / mcp_to_wrist
            ratios.append(ratio)

            if ratio > config.FINGER_EXTENDED_THRESHOLD:
                extended += 1
            elif ratio < config.FINGER_CURLED_THRESHOLD:
                curled += 1

        if extended >= config.MIN_FINGERS_FOR_PALM:
            state = HandState.PALM
        elif curled >= config.MIN_FINGERS_FOR_FIST:
            state = HandState.FIST
        else:
            state = HandState.UNKNOWN

        return DetectionDetail(
            state=state,
            finger_ratios=ratios,
            extended_count=extended,
            curled_count=curled,
            hands_found=len(results.multi_hand_landmarks),
            confidence=confidence,
            handedness=handedness,
            center_x=cx,
            center_y=cy,
            wrist_x=wrist.x,
            wrist_y=wrist.y,
        )

    @staticmethod
    def _dist(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def close(self):
        if self.hands:
            self.hands.close()
            logger.info("MediaPipe Hands closed")
