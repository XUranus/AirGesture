#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         config.py
#*   Author:       XUranus
#*   Date:         2026-03-14
#*   Description:  
#*
#================================================================*/
# /GrabDrop-Desktop/config.py
import uuid
import platform
import os

# Device
DEVICE_ID = uuid.uuid4().hex[:8]
DEVICE_NAME = platform.node() or "Desktop"

# Network
UDP_PORT = 9877
MULTICAST_GROUP = "239.255.77.88"
BROADCAST_TYPE_SCREENSHOT_READY = "SCREENSHOT_READY"
HEARTBEAT_TYPE = "HEARTBEAT"
SCREENSHOT_OFFER_TIMEOUT_S = 10
HEARTBEAT_INTERVAL_S = 3
DEVICE_TIMEOUT_S = 10

# Gesture Detection — Idle Stage
IDLE_FPS = 10
IDLE_FRAME_INTERVAL_S = 1.0 / IDLE_FPS
IDLE_WINDOW_SIZE = 10
IDLE_TRIGGER_THRESHOLD = 8

# Gesture Detection — Wakeup Stage
WAKEUP_FPS = 30
WAKEUP_FRAME_INTERVAL_S = 1.0 / WAKEUP_FPS
WAKEUP_DURATION_S = 2.0
WAKEUP_CONFIRM_FRAMES = 8

# Hand Classification
FINGER_EXTENDED_THRESHOLD = 1.3
FINGER_CURLED_THRESHOLD = 0.9
MIN_FINGERS_FOR_PALM = 3
MIN_FINGERS_FOR_FIST = 3

# Swipe Detection
SWIPE_DISPLACEMENT_THRESHOLD = 0.12
SWIPE_CONFIRM_FRAMES = 5
SWIPE_MIN_VELOCITY = 0.008
SWIPE_COOLDOWN_S = 0.8

# Cooldown
GRAB_COOLDOWN_S = 3.0
SWIPE_COOLDOWN_S = 0.8  # Shorter cooldown for swipe gestures

# Paths
SCREENSHOT_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "GrabDrop")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Overlay
OVERLAY_FONT_SIZE = 36
OVERLAY_DURATION_S = 2.0

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Camera Preview
SHOW_CAMERA_PREVIEW = True  # Set to False to disable preview window

# TCN Model Configuration
# Paths are relative to this config.py file
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BASE_DIR)

TCN_MODEL_CONFIG = os.path.join(_PROJECT_ROOT, "modelTraining", "checkpoints", "config.json")
TCN_MODEL_PATH = os.path.join(_PROJECT_ROOT, "modelTraining", "checkpoints", "gesture_tcn_best.pth")

# TCN Model Settings
USE_TCN_CLASSIFIER = True  # Set to False to use rule-based classifier
TCN_THRESHOLD = 0.5       # Confidence threshold
TCN_SMOOTH_WINDOW = 5     # Smoothing window size
TCN_WINDOW_SECONDS = 1.0  # Time window for gesture detection
TCN_FILL_FRAMES = 3       # Frames to fill when hand is lost
