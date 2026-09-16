#!/usr/bin/env python3
"""Run a small configurable experiment grid and save a CSV-like JSONL log."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["custom_cnn", "mobilenet_v2"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3, 3e-4])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-test-samples", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    log = args.output_dir / "results.jsonl"
    for model in args.models:
        for rate in args.learning_rates:
            run_name = f"{model}_lr_{rate:g}"
            output = args.output_dir / run_name
            command = [sys.executable, "-m", "traffic_sign_recognition.train", "--model", model, "--learning-rate", str(rate), "--epochs", str(args.epochs), "--max-train-samples", str(args.max_train_samples), "--max-test-samples", str(args.max_test_samples), "--no-pretrained", "--output-dir", str(output)]
            started = time.time(); completed = subprocess.run(command, check=False)
            config = json.loads((output / "config.json").read_text()) if (output / "config.json").exists() else {}
            with log.open("a") as file:
                file.write(json.dumps({"run": run_name, "model": model, "learning_rate": rate, "return_code": completed.returncode, "seconds": round(time.time() - started, 2), "test_accuracy": config.get("test_accuracy")}) + "\n")
    print(log)


if __name__ == "__main__": main()
