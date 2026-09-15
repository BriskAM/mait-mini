"""Evaluate a saved GTSRB model."""

import argparse
import json
from pathlib import Path
import tensorflow as tf
from .data import load_labeled_data, make_dataset
from .reporting import save_evaluation


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model-path", type=Path, required=True); p.add_argument("--data-dir", type=Path, default=Path("data/raw/gtsrb")); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--image-size", type=int, default=None)
    args = p.parse_args(); model = tf.keras.models.load_model(args.model_path)
    _, test = load_labeled_data(args.data_dir); size = args.image_size or model.input_shape[1]
    report = save_evaluation(model, make_dataset(test, size, args.batch_size), args.model_path.parent / "evaluation")
    print(json.dumps({"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]}, indent=2))


if __name__ == "__main__": main()
