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
IDLE_TRIGGER_THRESHOLD = 8  # 8 out of 10

# Gesture Detection — Wakeup Stage
WAKEUP_FPS = 30
WAKEUP_FRAME_INTERVAL_S = 1.0 / WAKEUP_FPS
WAKEUP_DURATION_S = 2.0
WAKEUP_CONFIRM_FRAMES = 8

# Hand Classification Thresholds
FINGER_EXTENDED_THRESHOLD = 1.3
FINGER_CURLED_THRESHOLD = 0.9
MIN_FINGERS_FOR_PALM = 3
MIN_FINGERS_FOR_FIST = 3

# Cooldown
GRAB_COOLDOWN_S = 3.0

# Paths
SCREENSHOT_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "GrabDrop")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Overlay
OVERLAY_FONT_SIZE = 36
OVERLAY_DURATION_S = 2.0

# Camera
CAMERA_INDEX = 0  # 0 = default webcam
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

