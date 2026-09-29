#!/usr/bin/env python3
"""
F1-macro vs minimum-confidence threshold, from the saved sparse predictions.

Uses the top-k predictions file produced by
validation/torchrun_save_predictions.py (values/indices/labels/files in one
.npz). No GPU and no model involved: everything is computed from the stored
top-1 prediction and its confidence.

Semantics of the threshold sweep: a top-1 prediction whose confidence is
below the threshold is NOT counted. It therefore contributes no TP and no
FP for the predicted class (e.g. a low-confidence wrong prediction no
longer penalizes the predicted class' precision). The image still has a
ground-truth class, so under the main metric the rejection counts as a miss
(FN) for the true class - the standard selective-classification view. At
threshold 0 nothing is rejected and the value equals the plain F1-macro of
the run.

For reference, a second curve "F1-macro (covered only)" excludes rejected
images entirely (neither FN nor FP), and "coverage" is the fraction of
images whose prediction is still counted.

F1-macro is computed over all classes (zero_division=0 for classes with no
TP/FP/FN), matching the f1_macro definition used during training.

Outputs (under validation/metrics/F1-macro/):
    plots/<predictions stem>_vs_threshold.png   - the plot
    <predictions stem>_threshold_sweep.csv      - the sweep data

Usage:
    python validation/metrics/F1-macro_threshold.py \
        [--predictions validation/predictions/convnextv2_base_top1000_predictions.npz] \
        [--step 0.005]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


# Editable defaults.
DEFAULT_PREDICTIONS = "validation/predictions/convnextv2_base_top1000_predictions.npz"
DEFAULT_METRICS_DIR = "validation/metrics/F1-macro"


def _resolve_path(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _load_predictions(predictions_path: Path):
    data = np.load(predictions_path)
    values = data["values"]
    indices = data["indices"]
    labels = data["labels"]
    return values, indices, labels


def _num_classes_from_metadata(predictions_path: Path, indices: np.ndarray, labels: np.ndarray) -> int:
    metadata_path = predictions_path.with_suffix("").with_suffix(".json")
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        num_classes = metadata.get("num_classes")
        if num_classes:
            return int(num_classes)
    return int(max(int(indices.max()), int(labels.max())) + 1)


def _sweep_thresholds(pred: np.ndarray, conf: np.ndarray, labels: np.ndarray, num_classes: int, thresholds: np.ndarray):
    """Compute per-threshold macro-F1 (rejection = miss), macro-F1 on the
    covered subset only, and coverage. Returns arrays aligned with thresholds."""
    n = len(labels)
    true_per_class = np.bincount(labels, minlength=num_classes)

    f1_macro = np.empty(len(thresholds), dtype=np.float64)
    f1_macro_covered = np.empty(len(thresholds), dtype=np.float64)
    coverage = np.empty(len(thresholds), dtype=np.float64)

    for i, t in enumerate(thresholds):
        counted = conf >= t
        correct = counted & (pred == labels)
        wrong = counted & ~correct

        tp = np.bincount(labels[correct], minlength=num_classes)
        fp = np.bincount(pred[wrong], minlength=num_classes)

        # Rejection counts as a miss for the true class.
        fn = true_per_class - tp
        denom = 2 * tp + fp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
        f1_macro[i] = f1.mean()

        # Covered-only variant: rejected images are excluded entirely.
        fn_covered = np.bincount(labels[wrong], minlength=num_classes)
        denom_covered = 2 * tp + fp + fn_covered
        with np.errstate(invalid="ignore", divide="ignore"):
            f1_covered = np.where(denom_covered > 0, 2 * tp / denom_covered, 0.0)
        f1_macro_covered[i] = f1_covered.mean()

        coverage[i] = counted.mean() if n else 0.0

    return f1_macro, f1_macro_covered, coverage


def _plot(thresholds, f1_macro, f1_macro_covered, coverage, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_i = int(np.argmax(f1_macro))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_macro, color="tab:blue", lw=2, label="F1-macro (rejected prediction = miss)")
    ax.plot(thresholds, f1_macro_covered, color="tab:orange", lw=1.5, ls="--", label="F1-macro (covered images only)")
    ax.plot(thresholds, coverage, color="tab:green", lw=1.5, ls=":", label="coverage (fraction of predictions counted)")
    ax.axvline(thresholds[best_i], color="tab:red", ls="--", lw=1)
    ax.annotate(
        f"best: t={thresholds[best_i]:.3f}\nF1-macro={f1_macro[best_i]:.4f}",
        xy=(thresholds[best_i], f1_macro[best_i]),
        xytext=(10, -30),
        textcoords="offset points",
        color="tab:red",
    )
    ax.set_xlabel("minimum confidence threshold (top-1 probability)")
    ax.set_ylabel("F1-macro / coverage")
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot F1-macro vs minimum confidence threshold from saved top-k predictions.")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS, help="Path to the sparse predictions .npz")
    parser.add_argument("--metrics-dir", default=DEFAULT_METRICS_DIR, help="Output directory; the plot goes to <metrics-dir>/plots")
    parser.add_argument("--step", type=float, default=0.005, help="Threshold step between 0 and 1 (default: 0.005)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    predictions_path = _resolve_path(repo_root, args.predictions)
    metrics_dir = _resolve_path(repo_root, args.metrics_dir)
    plots_dir = metrics_dir / "plots"

    if not predictions_path.exists():
        print(f"ERROR: predictions file not found: {predictions_path}", file=sys.stderr)
        print("Run validation/sbatch_save_predictions.sh first.", file=sys.stderr)
        return 1

    values, indices, labels = _load_predictions(predictions_path)
    num_classes = _num_classes_from_metadata(predictions_path, indices, labels)
    n = len(labels)

    pred = indices[:, 0].astype(np.int64)
    conf = values[:, 0].astype(np.float32)
    labels = labels.astype(np.int64)

    print(f"Predictions: {predictions_path}")
    print(f"Images: {n}, classes: {num_classes}")

    thresholds = np.round(np.arange(0.0, 1.0 + args.step / 2, args.step), 6)
    f1_macro, f1_macro_covered, coverage = _sweep_thresholds(pred, conf, labels, num_classes, thresholds)

    best_i = int(np.argmax(f1_macro))
    print(f"F1-macro at t=0 (no threshold): {f1_macro[0]:.4f}")
    print(f"Best F1-macro: {f1_macro[best_i]:.4f} at threshold {thresholds[best_i]:.3f} (coverage {coverage[best_i]:.4f})")

    stem = predictions_path.name
    for suffix in predictions_path.suffixes:
        stem = stem[: -len(suffix)]

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"{stem}_vs_threshold.png"
    title = (
        f"F1-macro vs min-confidence threshold\n"
        f"{predictions_path.name} ({n} images, {num_classes} classes)"
    )
    _plot(thresholds, f1_macro, f1_macro_covered, coverage, plot_path, title)
    print(f"Plot saved to {plot_path}")

    csv_path = metrics_dir / f"{stem}_threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "f1_macro", "f1_macro_covered", "coverage"])
        for row in zip(thresholds.tolist(), f1_macro.tolist(), f1_macro_covered.tolist(), coverage.tolist()):
            writer.writerow(row)
    print(f"Sweep data saved to {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
