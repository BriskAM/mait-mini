"""Train a GTSRB classifier."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import load_labeled_data, make_dataset, split_train_validation
from .models import build_model
from .reporting import save_evaluation, save_history


def set_seed(seed):
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["custom_cnn", "mobilenet_v2"], default="custom_cnn")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/gtsrb"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--image-size", type=int)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-test-samples", type=int)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args(); set_seed(args.seed)
    image_size = args.image_size or (96 if args.model == "mobilenet_v2" else 64)
    output = args.output_dir or Path("artifacts/models") / args.model
    train, test = load_labeled_data(args.data_dir)
    train, validation = split_train_validation(train, seed=args.seed)
    if args.max_train_samples: train = train.head(args.max_train_samples)
    if args.max_test_samples: test = test.head(args.max_test_samples)
    train_ds = make_dataset(train, image_size, args.batch_size, True, args.seed)
    val_ds = make_dataset(validation, image_size, args.batch_size)
    test_ds = make_dataset(test, image_size, args.batch_size)
    model = build_model(args.model, image_size, pretrained=not args.no_pretrained, seed=args.seed)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    output.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(output / "model.keras", monitor="val_accuracy", mode="max", save_best_only=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model.save(output / "model.keras")
    save_history(history, output)
    report = save_evaluation(model, test_ds, output / "evaluation")
    with (output / "config.json").open("w") as f:
        json.dump({"model": args.model, "image_size": image_size, "num_classes": 43, "seed": args.seed, "test_accuracy": report["accuracy"]}, f, indent=2)
    print(f"Saved model and reports under {output}")


if __name__ == "__main__": main()
