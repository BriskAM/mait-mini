"""Training and evaluation reports."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def save_history(history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.json").open("w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = list(range(1, len(history.history.get("accuracy", [])) + 1))
    for metric, ax in (("accuracy", axes[0]), ("loss", axes[1])):
        ax.plot(epochs, history.history.get(metric, []), marker="o", label="train")
        ax.plot(epochs, history.history.get("val_" + metric, []), marker="o", label="validation")
        ax.set_title(metric.title()); ax.set_xlabel("Epoch"); ax.set_xticks(epochs); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "training_curves.png", dpi=140); plt.close(fig)


def save_evaluation(model, dataset, output_dir: Path, class_names=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.concatenate([y.numpy() for _, y in dataset])
    y_prob = model.predict(dataset, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    names = class_names or [str(i) for i in range(model.output_shape[-1])]
    report = classification_report(y_true, y_pred, labels=list(range(len(names))), target_names=names, output_dict=True, zero_division=0)
    with (output_dir / "classification_report.json").open("w") as f: json.dump(report, f, indent=2)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    np.save(output_dir / "confusion_matrix.npy", cm)
    top_confusions = []
    for true_label in range(len(names)):
        for predicted_label in range(len(names)):
            if true_label != predicted_label and cm[true_label, predicted_label] > 0:
                top_confusions.append({"true_class": true_label, "predicted_class": predicted_label, "count": int(cm[true_label, predicted_label])})
    top_confusions.sort(key=lambda item: item["count"], reverse=True)
    with (output_dir / "top_confusions.json").open("w") as f: json.dump(top_confusions[:25], f, indent=2)
    fig, ax = plt.subplots(figsize=(12, 10)); im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("GTSRB confusion matrix")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6); ax.set_yticklabels(names, fontsize=6)
    fig.tight_layout(); fig.savefig(output_dir / "confusion_matrix.png", dpi=160); plt.close(fig)
    return report
