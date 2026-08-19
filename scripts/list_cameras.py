#!/usr/bin/env python3
"""List available OpenCV camera devices (helps find the Camo/Iriun index)."""
import sys
sys.path.insert(0, ".")
try:
    import cv2
    for i in range(5):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            ret, frame = cap.read()
            print(f"device {i}: OPEN, frame={'ok' if ret and frame is not None else 'empty'}")
        cap.release()
except ImportError:
    print("OpenCV not installed. pip install opencv-python")
