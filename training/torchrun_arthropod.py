#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


# Editable defaults.
DEFAULT_NPROC_PER_NODE = 2
DEFAULT_DATASET_DIR = "dataset"
DEFAULT_OUTPUT_DIR = "output/arthropod-classifier"


def _count_classes(class_map_path: Path) -> int:
    with class_map_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _build_command(repo_root: Path, dataset_dir: Path, output_dir: Path, nproc_per_node: int) -> list[str]:
    train_entry = repo_root / "src" / "train_arthropod.py"
    class_map = dataset_dir / "class-mapping.txt"

    if not train_entry.exists():
        raise FileNotFoundError(f"Missing train entrypoint: {train_entry}")
    if not class_map.exists():
        raise FileNotFoundError(f"Missing class map file: {class_map}")

    num_classes = _count_classes(class_map)
    if num_classes <= 0:
        raise RuntimeError(f"No classes found in class map: {class_map}")

    return [
        "torchrun",
        "--nproc_per_node",
        str(nproc_per_node),
        str(train_entry),
        "--data-dir",
        str(dataset_dir),
        "--model",
        "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "--pretrained",
        "--num-classes",
        str(num_classes),
        "--input-size",
        "3",
        "384",
        "384",
        "--class-map",
        str(class_map),
        "--epochs",
        "100",
        "-b",
        "32",
        "-vb",
        "64",
        "-j",
        "16",
        "--log-interval",
        "200",
        "--opt",
        "lamb",
        "--lr",
        "3e-4",
        "--sched",
        "cosine",
        "--weight-decay",
        "0.01",
        "--warmup-epochs",
        "5",
        "--smoothing",
        "0.1",
        "--drop-path",
        "0.05",
        "--mixup",
        "0.2",
        "--cutmix",
        "1.0",
        "--hflip",
        "0.5",
        "--aa",
        "rand-m7-mstd0.5",
        "--bce-loss",
        "--amp",
        "--eval-metric",
        "f1_macro",
        "--output",
        str(output_dir),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Modified timm training with torchrun.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Dataset directory containing train/val/test and class-mapping.txt")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for timm checkpoints/logs")
    parser.add_argument("--nproc-per-node", type=int, default=DEFAULT_NPROC_PER_NODE, help="Number of processes (GPUs) per node")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running it")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = (repo_root / args.dataset_dir).resolve() if not Path(args.dataset_dir).is_absolute() else Path(args.dataset_dir)
    output_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    cmd = _build_command(
        repo_root=repo_root,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        nproc_per_node=args.nproc_per_node,
    )
    print("$ " + shlex.join(cmd))

    if args.dry_run:
        return 0

    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
