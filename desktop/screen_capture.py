#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         screen_capture.py
#*   Author:       XUranus
#*   Date:         2026-03-14
#*   Description:  
#*
#================================================================*/

# /GrabDrop-Desktop/screen_capture.py
import logging
import os
import subprocess
import shutil
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger("ScreenCapture")


class ScreenCapture:
    """
    Screen capture with multiple backends.
    Tries in order:
      1. spectacle (KDE)
      2. grim (Wayland generic)
      3. gnome-screenshot (GNOME)
      4. scrot (X11)
      5. mss (X11 python)
      6. import (ImageMagick)
    """

    def __init__(self):
        self._backend = None
        self._detect_backend()

    def _detect_backend(self):
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()

        logger.info(
            f"Session: type={session_type} desktop={desktop} "
            f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', 'unset')} "
            f"DISPLAY={os.environ.get('DISPLAY', 'unset')}"
        )

        # Ordered preference based on desktop/session
        candidates = []

        if "kde" in desktop or "plasma" in desktop:
            candidates.append(("spectacle", self._capture_spectacle))

        if session_type == "wayland" or os.environ.get("WAYLAND_DISPLAY"):
            candidates.append(("grim", self._capture_grim))

        if "gnome" in desktop:
            candidates.append(("gnome-screenshot", self._capture_gnome))

        # Always add X11/generic fallbacks
        candidates.append(("scrot", self._capture_scrot))
        candidates.append(("mss", self._capture_mss))
        candidates.append(("import", self._capture_import))

        # Test each
        for name, func in candidates:
            if name == "mss":
                # mss doesn't need an external binary
                if self._test_mss():
                    self._backend = (name, func)
                    logger.info(f"✅ Using screenshot backend: {name}")
                    return
            else:
                binary = name
                if name == "gnome-screenshot":
                    binary = "gnome-screenshot"
                if shutil.which(binary):
                    self._backend = (name, func)
                    logger.info(f"✅ Using screenshot backend: {name}")
                    return
                else:
                    logger.debug(f"Backend {name}: binary not found")

        logger.error(
            "❌ No screenshot backend available! "
            "Install one of: spectacle, grim, scrot, gnome-screenshot"
        )
        self._backend = ("mss", self._capture_mss)  # last resort

    def _test_mss(self) -> bool:
        try:
            import mss
            with mss.mss() as sct:
                mon = sct.monitors[1]
                sct.grab(mon)
            logger.debug("mss test capture succeeded")
            return True
        except Exception as e:
            logger.debug(f"mss test failed: {e}")
            return False

    def capture(self) -> Optional[str]:
        """Capture screen and return saved file path."""
        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"GrabDrop_Sent_{timestamp}.png"
        filepath = os.path.join(config.SCREENSHOT_DIR, filename)

        if self._backend is None:
            logger.error("No screenshot backend configured")
            return None

        name, func = self._backend
        logger.debug(f"Capturing with backend: {name}")

        try:
            success = func(filepath)
            if success and os.path.exists(filepath):
                size = os.path.getsize(filepath)
                logger.info(
                    f"Screenshot saved: {filepath} ({size} bytes) "
                    f"[backend={name}]"
                )
                return filepath
            else:
                logger.error(f"Backend {name} returned no file")
                # Try fallback
                return self._try_fallbacks(filepath, name)
        except Exception as e:
            logger.error(
                f"Backend {name} failed: {e}", exc_info=True
            )
            return self._try_fallbacks(filepath, name)

    def _try_fallbacks(
        self, filepath: str, failed_backend: str
    ) -> Optional[str]:
        """Try all other backends if primary fails."""
        fallbacks = [
            ("spectacle", self._capture_spectacle),
            ("grim", self._capture_grim),
            ("gnome-screenshot", self._capture_gnome),
            ("scrot", self._capture_scrot),
            ("mss", self._capture_mss),
            ("import", self._capture_import),
        ]

        for name, func in fallbacks:
            if name == failed_backend:
                continue
            if name != "mss" and not shutil.which(
                name.split("-")[0] if "-" not in name else name
            ):
                continue

            try:
                logger.info(f"Trying fallback backend: {name}")
                success = func(filepath)
                if success and os.path.exists(filepath):
                    logger.info(
                        f"Fallback {name} succeeded! "
                        f"Switching default backend."
                    )
                    self._backend = (name, func)
                    return filepath
            except Exception as e:
                logger.debug(f"Fallback {name} failed: {e}")

        logger.error("All screenshot backends failed")
        return None

    # --- Backend Implementations ---

    def _capture_spectacle(self, filepath: str) -> bool:
        """KDE Spectacle — works on both X11 and Wayland."""
        result = subprocess.run(
            [
                "spectacle",
                "--background",       # no GUI
                "--nonotify",         # no notification
                "--fullscreen",       # full screen
                "--output", filepath,
            ],
            timeout=10,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.debug(
                f"spectacle stderr: {result.stderr.strip()}"
            )
        return result.returncode == 0

    def _capture_grim(self, filepath: str) -> bool:
        """grim — Wayland native screenshooter."""
        result = subprocess.run(
            ["grim", filepath],
            timeout=10,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.debug(f"grim stderr: {result.stderr.strip()}")
        return result.returncode == 0

    def _capture_gnome(self, filepath: str) -> bool:
        """gnome-screenshot — GNOME desktop."""
        result = subprocess.run(
            [
                "gnome-screenshot",
                "--file", filepath,
            ],
            timeout=10,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.debug(
                f"gnome-screenshot stderr: {result.stderr.strip()}"
            )
        return result.returncode == 0

    def _capture_scrot(self, filepath: str) -> bool:
        """scrot — X11 screenshot tool."""
        result = subprocess.run(
            ["scrot", filepath],
            timeout=10,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.debug(f"scrot stderr: {result.stderr.strip()}")
        return result.returncode == 0

    def _capture_import(self, filepath: str) -> bool:
        """ImageMagick import — X11."""
        result = subprocess.run(
            ["import", "-window", "root", filepath],
            timeout=10,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.debug(f"import stderr: {result.stderr.strip()}")
        return result.returncode == 0

    def _capture_mss(self, filepath: str) -> bool:
        """mss — Python X11 capture."""
        try:
            import mss
            from PIL import Image

            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                img = Image.frombytes(
                    "RGB",
                    (screenshot.width, screenshot.height),
                    screenshot.rgb,
                )
                img.save(filepath, "PNG")
            return True
        except Exception as e:
            logger.debug(f"mss capture error: {e}")
            return False

