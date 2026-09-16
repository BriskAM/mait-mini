"""Predict one traffic-sign image."""

import argparse
from pathlib import Path
import tensorflow as tf
from .inference import predict_image


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model-path", type=Path, required=True); p.add_argument("--image", type=Path, required=True); args = p.parse_args()
    model = tf.keras.models.load_model(args.model_path)
    result = predict_image(model, args.image)
    print(f"class_id={result['class_id']} confidence={result['confidence']:.4f}")
    print("top_k=" + ",".join(f"{row['class_id']}:{row['confidence']:.4f}" for row in result["top_k"]))


if __name__ == "__main__": main()
