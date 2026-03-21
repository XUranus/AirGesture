#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*   
#*   File:         main.py
#*   Author:       XUranus
#*   Date:         2026-03-16
#*   Description:  
#*
#================================================================*/

#!/usr/bin/env python3
"""gesture_capture/main.py — Entry point for Gesture Motion Capture."""

import sys
import argparse
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(
        description="Gesture Motion Capture — record and label hand-gesture video clips"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for the dataset (default: ./dataset)"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Gesture Motion Capture")

    window = MainWindow(camera_index=args.camera, output_dir=args.output)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

