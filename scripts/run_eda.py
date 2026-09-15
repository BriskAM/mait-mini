#!/usr/bin/env python3
"""Generate exploratory data analysis figures and summary JSON for GTSRB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from traffic_sign_recognition.data import load_labeled_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/gtsrb"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eda"))
    parser.add_argument("--website-dir", type=Path, default=Path("site/dist"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.website_dir / "assets").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    train, test = load_labeled_data(args.data_dir)

    counts = train.label.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(counts.index, counts.values, color="#24d6c8")
    ax.set(title="GTSRB training images by class", xlabel="Class ID", ylabel="Images")
    ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    fig.savefig(args.output_dir / "class_distribution.png", dpi=160); plt.close(fig)

    sample = train.iloc[rng.choice(len(train), size=min(25, len(train)), replace=False)]
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
        ax.imshow(Image.open(row.path).convert("RGB")); ax.set_title(f"Class {row.label}", fontsize=9); ax.axis("off")
    fig.suptitle("Random GTSRB training examples", y=0.995); fig.tight_layout()
    fig.savefig(args.output_dir / "sample_grid.png", dpi=160); plt.close(fig)

    dimensions = []
    brightness = []
    color_means = []
    for path in train.path.iloc[rng.choice(len(train), size=min(5000, len(train)), replace=False)]:
        image = np.asarray(Image.open(path).convert("RGB"))
        dimensions.append(image.shape[:2])
        brightness.append(float(image.mean()))
        color_means.append(image.mean(axis=(0, 1)))
    dimensions = np.asarray(dimensions); brightness = np.asarray(brightness); color_means = np.asarray(color_means)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist2d(dimensions[:, 1], dimensions[:, 0], bins=30, cmap="magma"); axes[0].set(title="Image dimensions", xlabel="Width", ylabel="Height")
    axes[1].hist(brightness, bins=40, color="#ffb454", edgecolor="white"); axes[1].set(title="Brightness distribution", xlabel="Mean pixel value", ylabel="Images")
    fig.tight_layout(); fig.savefig(args.output_dir / "image_statistics.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(color_means.mean(axis=0), marker="o", color="#8f7cff")
    ax.set_xticks([0, 1, 2], ["Red", "Green", "Blue"]); ax.set_ylim(0, 255)
    ax.set(title="Mean channel intensity across a 5,000-image sample", ylabel="Mean intensity")
    ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(args.output_dir / "color_statistics.png", dpi=160); plt.close(fig)

    for image in sorted(args.output_dir.glob("*.png")):
        target = args.website_dir / "assets" / image.name
        target.write_bytes(image.read_bytes())
    summary = {
        "train_images": int(len(train)), "test_images": int(len(test)), "classes": int(train.label.nunique()),
        "class_min": int(counts.min()), "class_max": int(counts.max()), "class_mean": float(counts.mean()),
        "sampled_dimensions": {"width_min": int(dimensions[:, 1].min()), "width_max": int(dimensions[:, 1].max()), "height_min": int(dimensions[:, 0].min()), "height_max": int(dimensions[:, 0].max())},
        "sampled_brightness_mean": float(brightness.mean()), "sampled_brightness_std": float(brightness.std()),
        "sampled_channel_means": [float(v) for v in color_means.mean(axis=0)],
    }
    for destination in (args.output_dir / "summary.json", args.website_dir / "data" / "eda_summary.json"):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
