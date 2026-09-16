#!/usr/bin/env python3
"""Optional YOLO traffic-sign detection entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an optional YOLO detector")
    parser.add_argument("--weights", required=True, help="YOLO detector checkpoint (.pt)")
    parser.add_argument("--source", required=True, help="Image, video, directory, or webcam index")
    parser.add_argument("--output", type=Path, default=Path("artifacts/detection"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install optional detection dependencies: uv pip install -r requirements-detection.txt") from exc
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz, device=device, save=True, project=args.output.parent, name=args.output.name)
    print(f"device={device} results={len(results)} output={args.output}")


if __name__ == "__main__":
    main()

