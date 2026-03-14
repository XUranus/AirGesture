#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         main.py
#*   Author:       XUranus
#*   Date:         2026-03-14
#*   Description:  
#*
#================================================================*/

# /GrabDrop-Desktop/main.py
import logging
import os
import sys
import signal
import time
import threading
from datetime import datetime
from collections import deque

import config
from gesture_detector import GestureDetector, GestureEvent
from screen_capture import ScreenCapture
from network_manager import NetworkManager
from overlay import OverlayManager
from sound_player import SoundPlayer


def setup_logging():
    os.makedirs(config.LOG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOG_DIR, f"grabdrop_{timestamp}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-5s] %(name)-18s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    logging.info(f"Log file: {log_file}")
    return log_file


class GrabDropDesktop:
    def __init__(self):
        self.logger = logging.getLogger("GrabDrop")
        self.running = False
        self._shutdown_event = threading.Event()

        self.gesture_detector = GestureDetector()
        self.screen_capture = ScreenCapture()
        self.network = NetworkManager()
        self.overlay = OverlayManager()
        self.sound = SoundPlayer()

        self.last_grab_time = 0.0

        # Offer queue instead of single offer
        self._offers_lock = threading.Lock()
        self._pending_offers = deque()  # list of offers with received_at

        # Track recent RELEASE events that had no offer
        # so we can match them retroactively
        self._unmatched_release_time = 0.0
        self._unmatched_release_lock = threading.Lock()

    def start(self):
        self.running = True
        self.logger.info("=" * 60)
        self.logger.info("  GrabDrop Desktop Client Starting")
        self.logger.info(f"  Device ID: {config.DEVICE_ID}")
        self.logger.info(f"  Device Name: {config.DEVICE_NAME}")
        self.logger.info(f"  Screenshots: {config.SCREENSHOT_DIR}")
        self.logger.info("=" * 60)

        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)

        self.overlay.start()

        self.network.start()
        self.network.on_screenshot_offer = self._on_incoming_offer

        self.gesture_detector.on_gesture = self._on_gesture
        self.gesture_detector.on_stage_change = self._on_stage_change
        self.gesture_detector.start()

        self.logger.info("All components started. Waiting for gestures...")
        self.logger.info("Press Ctrl+C to stop.")

        try:
            self._run_with_interrupt_check()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _run_with_interrupt_check(self):
        if self.overlay.is_available():
            root = self.overlay.get_root()
            if root:
                def check_shutdown():
                    if self._shutdown_event.is_set():
                        root.quit()
                        return
                    root.after(200, check_shutdown)

                root.after(200, check_shutdown)

                try:
                    root.mainloop()
                except KeyboardInterrupt:
                    self.logger.info("KeyboardInterrupt in mainloop")
            else:
                self._blocking_wait()
        else:
            self._blocking_wait()

    def _blocking_wait(self):
        try:
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=0.5)
        except KeyboardInterrupt:
            pass

    def request_shutdown(self):
        self._shutdown_event.set()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._shutdown_event.set()
        self.logger.info("Shutting down...")

        self.gesture_detector.stop()
        self.network.stop()
        self.overlay.stop()
        self.sound.cleanup()

        self.logger.info("GrabDrop Desktop stopped.")

    # --- Offer Management ---

    def _add_offer(self, offer: dict):
        """Add offer to queue with local received timestamp."""
        offer["received_at"] = time.time()

        with self._offers_lock:
            self._pending_offers.append(offer)
            # Clean old offers
            self._cleanup_expired_offers()
            count = len(self._pending_offers)

        self.logger.info(
            f"📡 Offer queued from {offer['sender_name']} "
            f"({offer['sender_address']}:{offer['tcp_port']}) "
            f"— {count} offer(s) pending"
        )

    def _get_best_offer(self) -> dict | None:
        """Get the freshest non-expired offer."""
        with self._offers_lock:
            self._cleanup_expired_offers()

            if not self._pending_offers:
                return None

            # Take the most recent offer
            offer = self._pending_offers.pop()
            # Clear remaining (use only latest)
            self._pending_offers.clear()
            return offer

    def _cleanup_expired_offers(self):
        """Remove expired offers from the queue. Must hold lock."""
        now = time.time()
        while self._pending_offers:
            oldest = self._pending_offers[0]
            age = now - oldest["received_at"]
            if age > config.SCREENSHOT_OFFER_TIMEOUT_S:
                expired = self._pending_offers.popleft()
                self.logger.debug(
                    f"Offer expired from {expired['sender_name']} "
                    f"(age={age:.1f}s)"
                )
            else:
                break

    def _has_pending_offers(self) -> bool:
        with self._offers_lock:
            self._cleanup_expired_offers()
            return len(self._pending_offers) > 0

    # --- Event Handlers ---

    def _on_gesture(self, event: GestureEvent):
        if event == GestureEvent.GRAB:
            self._handle_grab()
        elif event == GestureEvent.RELEASE:
            self._handle_release()

    def _on_stage_change(self, stage: str, indicator: str):
        if stage == "WAKEUP":
            self.overlay.show_wakeup_indicator(indicator)
        else:
            self.overlay.hide_wakeup_indicator()

    def _handle_grab(self):
        now = time.time()
        if now - self.last_grab_time < config.GRAB_COOLDOWN_S:
            self.logger.info("GRAB ignored (cooldown)")
            return
        self.last_grab_time = now

        self.logger.info("✊ GRAB — Taking screenshot...")

        self.overlay.show_flash()
        self.sound.play_shutter()

        filepath = self.screen_capture.capture()
        if filepath is None:
            self.logger.error("❌ Screenshot capture failed")
            return

        self.logger.info(f"📸 Screenshot saved: {filepath}")
        self.overlay.show_thumbnail(filepath)

        try:
            with open(filepath, "rb") as f:
                data = f.read()
            self.network.broadcast_screenshot(data)
            self.logger.info(f"📡 Broadcast sent ({len(data)} bytes)")
        except Exception as e:
            self.logger.error(f"Broadcast failed: {e}")

    def _handle_release(self):
        offer = self._get_best_offer()

        if offer is None:
            self.logger.info(
                "🤚 RELEASE detected — no pending offer, "
                "recording for retroactive match"
            )
            # Record this unmatched RELEASE for retroactive matching
            with self._unmatched_release_lock:
                self._unmatched_release_time = time.time()
            return

        age = time.time() - offer["received_at"]
        self.logger.info(
            f"🤚 RELEASE — matched offer from {offer['sender_name']} "
            f"(age={age:.1f}s)"
        )

        self._download_and_save(offer)

    def _on_incoming_offer(self, offer: dict):
        self._add_offer(offer)

        # Check if there was a recent unmatched RELEASE
        # (RELEASE came before the offer — retroactive match)
        with self._unmatched_release_lock:
            release_age = time.time() - self._unmatched_release_time
            had_recent_release = release_age < 3.0  # within 3 seconds
            if had_recent_release:
                self._unmatched_release_time = 0.0  # consume it

        if had_recent_release:
            self.logger.info(
                f"🔄 Retroactive match! RELEASE was {release_age:.1f}s ago "
                f"— auto-accepting offer from {offer['sender_name']}"
            )
            # Process in a separate thread
            matched_offer = self._get_best_offer()
            if matched_offer:
                threading.Thread(
                    target=self._download_and_save,
                    args=(matched_offer,),
                    daemon=True,
                ).start()

    def _download_and_save(self, offer: dict):
        """Download screenshot from offer and save."""
        self.logger.info(
            f"📥 Downloading from {offer['sender_name']} "
            f"({offer['sender_address']}:{offer['tcp_port']})..."
        )

        data = self.network.download_screenshot(offer)
        if data is None:
            self.logger.error("❌ Download failed")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"GrabDrop_Received_{timestamp}.png"
        filepath = os.path.join(config.SCREENSHOT_DIR, filename)

        try:
            with open(filepath, "wb") as f:
                f.write(data)
            self.logger.info(f"📥 Received & saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            return

        self.overlay.show_ripple()
        self.sound.play_receive()
        self._open_image(filepath)

    def _open_image(self, filepath: str):
        try:
            import subprocess
            import platform as plat

            system = plat.system()
            if system == "Darwin":
                subprocess.Popen(["open", filepath])
            elif system == "Windows":
                os.startfile(filepath)
            else:
                subprocess.Popen(["xdg-open", filepath])
        except Exception as e:
            self.logger.error(f"Failed to open image: {e}")


_app: GrabDropDesktop = None


def main():
    global _app

    log_file = setup_logging()

    _app = GrabDropDesktop()

    def signal_handler(sig, frame):
        logging.info("Interrupt received, requesting shutdown...")
        if _app:
            _app.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    _app.start()


if __name__ == "__main__":
    main()

