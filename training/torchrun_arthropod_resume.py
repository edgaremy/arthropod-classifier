#!/usr/bin/env python3
"""
Torchrun launcher for arthropod classifier training with checkpoint resume support.

When resuming from a checkpoint, this script automatically:
1. Extracts the checkpoint epoch from the filename (e.g., 'checkpoint-30.pth.tar' -> 30)
2. Sets --start-epoch to that epoch value

This ensures the cosine scheduler fast-forwards to the correct epoch and calculates
the LR correctly based on the cosine decay formula. The warmup settings should match
the original training (default: --warmup-epochs 5, --warmup-lr 1e-05).

This workaround is necessary because timm checkpoints do not save lr_scheduler state,
but the scheduler's LR calculation is deterministic, so fast-forwarding with --start-epoch
should restore the correct LR as long as all other scheduler parameters match.

Usage:
    python torchrun_arthropod_resume.py --checkpoint <path_to_checkpoint>
    python torchrun_arthropod_resume.py --checkpoint <path> --dry-run
"""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path


# Editable defaults.
DEFAULT_NPROC_PER_NODE = 2
DEFAULT_DATASET_DIR = "dataset"
DEFAULT_OUTPUT_DIR = "output/arthropod-classifier"
DEFAULT_CHECKPOINT = ""
DEFAULT_RESUME = True  # Resume full model and optimizer state from checkpoint[]


def _count_classes(class_map_path: Path) -> int:
    with class_map_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _get_last_lr_from_summary(summary_path: Path) -> float:
    """Extract the last learning rate from a timm summary.csv file."""
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_row = rows[-1]
                lr_str = last_row.get("lr", "").strip()
                if lr_str:
                    return float(lr_str)
    except Exception:
        pass
    return None


def _get_checkpoint_epoch(checkpoint_path: str) -> int:
    """Extract epoch number from checkpoint filename (e.g., 'checkpoint-30.pth.tar' -> 30)."""
    import re
    match = re.search(r'checkpoint-(\d+)\.', checkpoint_path)
    if match:
        return int(match.group(1))
    return None


def _build_command(
    repo_root: Path,
    dataset_dir: Path,
    output_dir: Path,
    nproc_per_node: int,
    checkpoint: str = "",
    resume: bool = False,
    warmup_lr: float | None = None,
    warmup_epochs: int | None = None,
    start_epoch: int | None = None,
) -> list[str]:
    train_entry = repo_root / "src" / "train_arthropod.py"
    class_map = dataset_dir / "class-mapping.txt"

    if not train_entry.exists():
        raise FileNotFoundError(f"Missing train entrypoint: {train_entry}")
    if not class_map.exists():
        raise FileNotFoundError(f"Missing class map file: {class_map}")

    num_classes = _count_classes(class_map)
    if num_classes <= 0:
        raise RuntimeError(f"No classes found in class map: {class_map}")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
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

    if resume and checkpoint:
        cmd.extend(["--resume", checkpoint])
    elif checkpoint:
        cmd.extend(["--initial-checkpoint", checkpoint])

    if warmup_lr is not None:
        cmd.extend(["--warmup-lr", str(warmup_lr)])

    if warmup_epochs is not None:
        # Find and replace the default --warmup-epochs 5
        for i, arg in enumerate(cmd):
            if arg == "--warmup-epochs":
                cmd[i + 1] = str(warmup_epochs)
                break
        else:
            cmd.extend(["--warmup-epochs", str(warmup_epochs)])

    if start_epoch is not None:
        cmd.extend(["--start-epoch", str(start_epoch)])

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Modified timm training with torchrun.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Dataset directory containing train/val/test and class-mapping.txt")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for timm checkpoints/logs")
    parser.add_argument("--nproc-per-node", type=int, default=DEFAULT_NPROC_PER_NODE, help="Number of processes (GPUs) per node")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to checkpoint file")
    parser.add_argument("--resume", action="store_true", default=DEFAULT_RESUME, help="Resume training from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running it")
    parser.add_argument(
        "--no-auto-warmup-lr",
        action="store_true",
        help="Disable automatic warmup-lr detection from checkpoint summary",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = (repo_root / args.dataset_dir).resolve() if not Path(args.dataset_dir).is_absolute() else Path(args.dataset_dir)
    output_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    checkpoint_path = (repo_root / args.checkpoint).resolve() if args.checkpoint and not Path(args.checkpoint).is_absolute() else Path(args.checkpoint) if args.checkpoint else ""
    cpu_threads = os.cpu_count() or 1
    omp_threads = max(1, cpu_threads // args.nproc_per_node)
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    # Try to extract the checkpoint epoch from the filename for --start-epoch
    start_epoch_override = None
    if args.resume and checkpoint_path:
        checkpoint_dir = checkpoint_path.parent
        
        # Extract checkpoint epoch from filename
        checkpoint_epoch = _get_checkpoint_epoch(str(checkpoint_path))
        if checkpoint_epoch is not None:
            start_epoch_override = checkpoint_epoch
            print(f"Auto-detected checkpoint epoch: {start_epoch_override}")
            print("Using --start-epoch to ensure scheduler fast-forwards correctly")
        else:
            print("Warning: Could not extract epoch from checkpoint filename")
    
    # Note: We no longer set --warmup-lr or override --warmup-epochs
    # The scheduler will fast-forward to start_epoch and calculate LR correctly
    # as long as warmup_epochs matches the original training (5)

    cmd = _build_command(
        repo_root=repo_root,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        nproc_per_node=args.nproc_per_node,
        checkpoint=str(checkpoint_path) if checkpoint_path else "",
        resume=args.resume,
        warmup_lr=None,
        warmup_epochs=None,
        start_epoch=start_epoch_override,
    )
    print("$ " + shlex.join(cmd))

    if args.dry_run:
        return 0

    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
