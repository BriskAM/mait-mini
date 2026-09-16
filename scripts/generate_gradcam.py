#!/usr/bin/env python3
"""Generate a Grad-CAM heatmap for one image."""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from traffic_sign_recognition.gradcam import explain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/explanations/gradcam.png"))
    args = parser.parse_args()
    result = explain(tf.keras.models.load_model(args.model_path), args.image, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
