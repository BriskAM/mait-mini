#!/usr/bin/env python3
"""Build comparison graphics and website data from completed training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--website-dir", type=Path, default=Path("site/dist"))
    args = parser.parse_args()
    args.website_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_dir in sorted(args.models_dir.iterdir() if args.models_dir.exists() else []):
        if not model_dir.is_dir():
            continue
        config = read_json(model_dir / "config.json")
        if not config:
            continue
        report = read_json(model_dir / "evaluation" / "classification_report.json")
        name = config.get("model", model_dir.name)
        rows.append({"name": name, "accuracy": float(config.get("test_accuracy", report.get("accuracy", 0))), "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0)), "path": str(model_dir)})
        assets = args.website_dir / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        for source_name, target_name in (("training_curves.png", f"{name}_training_curves.png"), ("evaluation/confusion_matrix.png", f"{name}_confusion_matrix.png")):
            source = model_dir / source_name
            if source.exists():
                (assets / target_name).write_bytes(source.read_bytes())
    if rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [r["name"].replace("_", " ").title() for r in rows]
        x = range(len(rows)); width = 0.36
        ax.bar([i - width / 2 for i in x], [r["accuracy"] for r in rows], width, label="Accuracy", color="#24d6c8")
        ax.bar([i + width / 2 for i in x], [r["macro_f1"] for r in rows], width, label="Macro F1", color="#8f7cff")
        ax.set_xticks(list(x), labels); ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Model comparison")
        ax.grid(axis="y", alpha=0.2); ax.legend(); fig.tight_layout()
        fig.savefig(args.models_dir / "model_comparison.png", dpi=160); plt.close(fig)
        (args.website_dir / "assets" / "model_comparison.png").write_bytes((args.models_dir / "model_comparison.png").read_bytes())
    payload = {"models": rows}
    (args.website_dir / "data").mkdir(parents=True, exist_ok=True)
    (args.website_dir / "data" / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
