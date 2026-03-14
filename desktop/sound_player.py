#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         sound_player.py
#*   Author:       XUranus
#*   Date:         2026-03-14
#*   Description:  
#*
#================================================================*/

# /GrabDrop-Desktop/sound_player.py
import logging
import platform
import subprocess
import threading
from typing import Optional

logger = logging.getLogger("Sound")


class SoundPlayer:
    """
    Cross-platform simple sound effects.
    Uses system beep / afplay / aplay as fallback.
    """

    def __init__(self):
        self._system = platform.system()
        logger.info(f"Sound player initialized (platform={self._system})")

    def play_shutter(self):
        threading.Thread(target=self._do_shutter, daemon=True).start()

    def play_receive(self):
        threading.Thread(target=self._do_receive, daemon=True).start()

    def _do_shutter(self):
        try:
            if self._system == "Darwin":
                # macOS system sound
                subprocess.run(
                    [
                        "afplay",
                        "/System/Library/Sounds/Tink.aiff",
                    ],
                    timeout=2,
                    capture_output=True,
                )
            elif self._system == "Windows":
                import winsound
                winsound.MessageBeep(winsound.MB_OK)
            else:
                # Linux — try paplay, then aplay, then bell
                self._linux_beep(frequency=1200, duration_ms=100)
        except Exception as e:
            logger.debug(f"Shutter sound failed: {e}")

    def _do_receive(self):
        try:
            if self._system == "Darwin":
                subprocess.run(
                    [
                        "afplay",
                        "/System/Library/Sounds/Pop.aiff",
                    ],
                    timeout=2,
                    capture_output=True,
                )
            elif self._system == "Windows":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
            else:
                self._linux_beep(frequency=800, duration_ms=150)
        except Exception as e:
            logger.debug(f"Receive sound failed: {e}")

    def _linux_beep(self, frequency=1000, duration_ms=100):
        """Generate beep on Linux using available tools."""
        try:
            # Try paplay with generated audio
            subprocess.run(
                [
                    "paplay",
                    "/usr/share/sounds/freedesktop/stereo/camera-shutter.oga",
                ],
                timeout=2,
                capture_output=True,
            )
            return
        except Exception:
            pass

        try:
            # Try beep command
            subprocess.run(
                [
                    "beep",
                    "-f", str(frequency),
                    "-l", str(duration_ms),
                ],
                timeout=2,
                capture_output=True,
            )
            return
        except Exception:
            pass

        # Terminal bell as last resort
        print("\a", end="", flush=True)

    def cleanup(self):
        pass

