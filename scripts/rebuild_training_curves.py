#!/usr/bin/env python3
"""Regenerate readable learning-curve plots from saved JSON histories."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    root = Path("artifacts/models")
    website_assets = Path("site/dist/assets")
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        history_path = model_dir / "history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text())
        epochs = list(range(1, len(history.get("accuracy", [])) + 1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for metric, ax in (("accuracy", axes[0]), ("loss", axes[1])):
            ax.plot(epochs, history.get(metric, []), marker="o", label="train")
            ax.plot(epochs, history.get("val_" + metric, []), marker="o", label="validation")
            ax.set_title(metric.title()); ax.set_xlabel("Epoch"); ax.set_xticks(epochs); ax.legend(); ax.grid(alpha=0.2)
        fig.tight_layout()
        output = model_dir / "training_curves.png"
        fig.savefig(output, dpi=140)
        plt.close(fig)
        website_assets.mkdir(parents=True, exist_ok=True)
        (website_assets / f"{model_dir.name}_training_curves.png").write_bytes(output.read_bytes())
        print(output)


if __name__ == "__main__":
    main()
