"""Predict one traffic-sign image."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model-path", type=Path, required=True); p.add_argument("--image", type=Path, required=True); args = p.parse_args()
    model = tf.keras.models.load_model(args.model_path)
    image = Image.open(args.image).convert("RGB").resize((model.input_shape[1], model.input_shape[2]))
    probs = model.predict(np.asarray(image, dtype="float32")[None] / 255.0, verbose=0)[0]
    idx = int(np.argmax(probs)); print(f"class_id={idx} confidence={probs[idx]:.4f}")


if __name__ == "__main__": main()
