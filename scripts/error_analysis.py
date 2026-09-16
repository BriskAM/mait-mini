#!/usr/bin/env python3
"""Summarize the most common class confusions from saved evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    args = parser.parse_args()
    pairs = json.loads((args.evaluation_dir / "top_confusions.json").read_text())
    pairs = pairs[:10]
    output = args.evaluation_dir / "top_confusions.png"
    if pairs:
        labels = [f"{p['true_class']} → {p['predicted_class']}" for p in pairs][::-1]
        values = [p["count"] for p in pairs][::-1]
        fig, ax = plt.subplots(figsize=(8, 4.5)); ax.barh(labels, values, color="#8f7cff")
        ax.set_xlabel("Misclassified test images"); ax.set_title("Most common class confusions"); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)
    print(json.dumps({"top_confusions": pairs, "plot": str(output) if pairs else None}, indent=2))


if __name__ == "__main__": main()
