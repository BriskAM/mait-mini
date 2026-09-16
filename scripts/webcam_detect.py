#!/usr/bin/env python3
"""Optional webcam YOLO detection demo using Apple MPS when available."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()
    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements-detection.txt plus OpenCV before webcam detection") from exc
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(args.weights)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    print(f"webcam detection device={device}; press q to quit")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            annotated = model.predict(frame, conf=args.conf, device=device, verbose=False)[0].plot()
            cv2.imshow("Traffic-sign detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

