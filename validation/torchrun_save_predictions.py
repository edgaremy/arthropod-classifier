#!/usr/bin/env python3
"""
Save top-k sparse predictions of the best arthropod-classifier checkpoint.

Instead of re-running the model for every analysis, this stores, for each
image of the test set, the top-k (default k=1000) class probabilities as a
sparse row, plus the ground-truth label and the image file path. Storing all
24k probabilities per image would be a ~104 GB dense matrix; the top-k tail
is numerically irrelevant, so top-1000 captures essentially all of the
softmax mass in ~4-5 GB.

Output format (single compressed .npz, plus a .json metadata sidecar):
    values   (N, k) float16  - top-k probabilities per row, sorted desc.
    indices  (N, k) uint16   - class indices of those probabilities
                               (int32 if num_classes > 65535)
    labels   (N,)    uint16   - ground-truth class index per image
    files    (N,)    <U      - image path relative to the split directory
Rows follow the eval dataloader order, which is sequential (no shuffle), so
row i corresponds to dataset.filenames()[i]; this is asserted at save time.

Two modes in one file, mirroring validation/torchrun_validation.py:

1. Launcher mode (default, no --entry): builds a torchrun command that
   re-invokes this same file in entry mode. Use it for local runs:

       python validation/torchrun_save_predictions.py [--dry-run]

2. Entry mode (--entry): loads the vendored timm validate.py to reuse its
   argument parser and model/dataset/loader setup (identical preprocessing
   to the metrics validation), then runs a capture loop instead of timm's
   metrics loop. This is what the launcher and
   validation/sbatch_save_predictions.sh execute.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.util
import json
import logging
import os
import shlex
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np


# Editable defaults.
DEFAULT_NPROC_PER_NODE = 1
DEFAULT_DATASET_DIR = "dataset"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL = "convnextv2_base.fcmae_ft_in22k_in1k_384"
DEFAULT_CHECKPOINT = "output/arthropod-classifier/20260926-234704-convnextv2_base_fcmae_ft_in22k_in1k_384-384/checkpoint-98.pth.tar"
# Path of the output predictions file (.npz). A .json metadata sidecar with
# the same stem is written next to it.
DEFAULT_PREDICTIONS = "validation/predictions/test_checkpoint-98_top1000_predictions.npz"
DEFAULT_TOP_K = 1000
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 16
DEFAULT_INPUT_SIZE = ["3", "384", "384"]

_logger = logging.getLogger("save_predictions")


def _count_classes(class_map_path: Path) -> int:
    with class_map_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _resolve_path(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _build_command(
    repo_root: Path,
    dataset_dir: Path,
    checkpoint: Path,
    predictions: Path,
    top_k: int,
    nproc_per_node: int,
    batch_size: int,
    workers: int,
) -> list[str]:
    class_map = dataset_dir / "class-mapping.txt"

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not class_map.exists():
        raise FileNotFoundError(f"Missing class map file: {class_map}")

    num_classes = _count_classes(class_map)
    if num_classes <= 0:
        raise RuntimeError(f"No classes found in class map: {class_map}")
    if top_k > num_classes:
        raise ValueError(f"top-k ({top_k}) cannot exceed num_classes ({num_classes})")

    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(nproc_per_node),
        str(Path(__file__).resolve()),
        "--entry",
        "--data-dir",
        str(dataset_dir),
        "--split",
        DEFAULT_SPLIT,
        "--model",
        DEFAULT_MODEL,
        "--num-classes",
        str(num_classes),
        "--input-size",
        *DEFAULT_INPUT_SIZE,
        "--class-map",
        str(class_map),
        "--checkpoint",
        str(checkpoint),
        "-b",
        str(batch_size),
        "-j",
        str(workers),
        "--amp",
        "--top-k",
        str(top_k),
        "--predictions",
        str(predictions),
    ]


def _load_timm_validate_module(repo_root: Path):
    # Reuse timm's validate.py: its argument parser and its model/dataset/
    # loader setup guarantee preprocessing identical to the metrics run.
    timm_repo = repo_root / "pytorch-image-models"
    validate_script = timm_repo / "validate.py"

    if not validate_script.exists():
        raise FileNotFoundError(f"timm validate.py not found at: {validate_script}")

    if str(timm_repo) not in sys.path:
        sys.path.insert(0, str(timm_repo))

    spec = importlib.util.spec_from_file_location("arthropod_timm_validate", validate_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for: {validate_script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _save_predictions(validate_module, args, top_k: int, predictions_path: Path) -> None:
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    amp_autocast = contextlib.suppress
    if args.amp:
        assert args.amp_dtype in ("float16", "bfloat16")
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        _logger.info("Predicting in mixed precision with native PyTorch AMP.")
    else:
        _logger.info("Predicting in float32. AMP not enabled.")

    if args.checkpoint:
        args.pretrained = False
    in_chans = args.in_chans if args.in_chans is not None else (args.input_size[0] if args.input_size else 3)

    model = validate_module.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=in_chans,
        global_pool=args.gp,
    )
    if args.num_classes is None:
        assert hasattr(model, "num_classes"), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = model.num_classes

    if not args.checkpoint:
        raise RuntimeError("A --checkpoint is required to save predictions.")

    validate_module.load_checkpoint(model, args.checkpoint, args.use_ema)
    model = model.to(device=device)
    model.eval()
    param_count = sum(m.numel() for m in model.parameters())
    _logger.info("Model %s created, param count: %d" % (args.model, param_count))

    data_config = validate_module.resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )

    root_dir = args.data or args.data_dir
    input_img_mode = args.input_img_mode
    if input_img_mode is None:
        input_img_mode = "RGB" if data_config["input_size"][0] == 3 else "L"
    dataset = validate_module.create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        download=args.dataset_download,
        load_bytes=args.tf_preprocessing,
        class_map=args.class_map,
        num_samples=args.num_samples,
        input_key=args.input_key,
        input_img_mode=input_img_mode,
        target_key=args.target_key,
        trust_remote_code=args.dataset_trust_remote_code,
        seed=args.seed,
    )

    loader = validate_module.create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        use_prefetcher=not args.no_prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=data_config["crop_pct"],
        crop_mode=data_config["crop_mode"],
        crop_border_pixels=args.crop_border_pixels,
        pin_memory=args.pin_mem,
        device=device,
        img_dtype=torch.float32,
        tf_preprocessing=args.tf_preprocessing,
    )

    num_classes = args.num_classes
    k = min(top_k, num_classes)
    index_dtype = np.uint16 if num_classes <= np.iinfo(np.uint16).max + 1 else np.int32
    _logger.info(
        "Saving top-%d of %d class probabilities for %d images (index dtype: %s)",
        k, num_classes, len(dataset), np.dtype(index_dtype).name,
    )

    values_chunks: list[np.ndarray] = []
    indices_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    num_seen = 0

    with torch.inference_mode():
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.to(device=device)
                input = input.to(device=device)

            with amp_autocast():
                output = model(input)

            # Softmax in float32 for numerical stability, then keep top-k.
            probs = output.float().softmax(dim=-1)
            topk_values, topk_indices = probs.topk(k, dim=1)

            values_chunks.append(topk_values.cpu().numpy().astype(np.float16))
            indices_chunks.append(topk_indices.cpu().numpy().astype(index_dtype))
            labels_chunks.append(target.detach().cpu().numpy().astype(index_dtype))
            num_seen += input.shape[0]

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    "Predict: [%4d/%d]  images so far: %d",
                    batch_idx, len(loader), num_seen,
                )

    if num_seen != len(dataset):
        raise RuntimeError(
            f"Dataloader yielded {num_seen} images but dataset has {len(dataset)}; "
            "refusing to save misaligned predictions."
        )

    values = np.concatenate(values_chunks) if values_chunks else np.empty((0, k), dtype=np.float16)
    indices = np.concatenate(indices_chunks) if indices_chunks else np.empty((0, k), dtype=index_dtype)
    labels = np.concatenate(labels_chunks) if labels_chunks else np.empty((0,), dtype=index_dtype)
    files = np.asarray(dataset.filenames())

    top1 = float(np.mean(indices[:, 0] == labels)) if num_seen else 0.0
    _logger.info("Sanity check - top-1 accuracy from saved rows: %.4f", top1)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    _logger.info("Writing %s", predictions_path)
    np.savez_compressed(
        predictions_path,
        values=values,
        indices=indices,
        labels=labels,
        files=files,
    )

    metadata = {
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_dir": str(root_dir),
        "split": args.split,
        "class_map": str(args.class_map),
        "num_classes": int(num_classes),
        "num_images": int(num_seen),
        "top_k": int(k),
        "values_dtype": "float16",
        "indices_dtype": np.dtype(index_dtype).name,
        "probabilities": "softmax over all classes, top-k kept per row, rows sorted descending",
        "row_order": "sequential eval dataloader order; row i == files[i] == labels[i]",
        "created": datetime.datetime.now().isoformat(),
    }
    metadata_path = predictions_path.with_suffix("").with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    _logger.info("Metadata written to %s", metadata_path)


def _run_save_predictions_entry(passthrough: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    validate_module = _load_timm_validate_module(repo_root)

    # Parse our extra flags out of the passthrough, then hand the rest to
    # timm validate.py's parser so every other argument (and its defaults)
    # behaves exactly as in the metrics validation run.
    save_parser = argparse.ArgumentParser(add_help=False)
    save_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of top probabilities to store per image")
    save_parser.add_argument("--predictions", default="", metavar="PATH", help="Output .npz path for the sparse predictions (default: validation/predictions/)")
    save_args, validate_passthrough = save_parser.parse_known_args(passthrough)

    args = validate_module.parser.parse_args(validate_passthrough)

    predictions_path = (
        _resolve_path(repo_root, save_args.predictions)
        if save_args.predictions
        else _default_predictions_path(repo_root, args.checkpoint, save_args.top_k)
    )
    _save_predictions(validate_module, args, save_args.top_k, predictions_path)
    return 0


def _default_predictions_path(repo_root: Path, checkpoint: str, top_k: int) -> Path:
    if not checkpoint:
        raise RuntimeError("A --checkpoint is required (pass --predictions for the output path).")
    checkpoint_path = _resolve_path(repo_root, checkpoint)
    stem = checkpoint_path.name
    for suffix in checkpoint_path.suffixes:
        stem = stem[: -len(suffix)]
    return repo_root / "validation" / "predictions" / f"test_{stem}_top{top_k}_predictions.npz"


def _parse_wrapper_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--entry",
        action="store_true",
        help="Run in entry mode: save predictions in-process (used by the launcher and sbatch_save_predictions.sh).",
    )
    return parser.parse_known_args(argv)


def main() -> int:
    wrapper_args, passthrough = _parse_wrapper_args(sys.argv[1:])

    if wrapper_args.entry:
        return _run_save_predictions_entry(passthrough)

    parser = argparse.ArgumentParser(description="Launch top-k prediction saving with torchrun.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Dataset directory containing train/val/test and class-mapping.txt")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to the checkpoint to predict with")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS, help="Output .npz path for the sparse predictions")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of top probabilities to store per image")
    parser.add_argument("--nproc-per-node", type=int, default=DEFAULT_NPROC_PER_NODE, help="Number of processes per node (prediction saving is single-process, keep 1)")
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("-j", "--workers", type=int, default=DEFAULT_WORKERS, help="Number of data loading workers")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running it")
    args = parser.parse_args(passthrough)

    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = _resolve_path(repo_root, args.dataset_dir)
    checkpoint = _resolve_path(repo_root, args.checkpoint)
    predictions = _resolve_path(repo_root, args.predictions)
    cpu_threads = os.cpu_count() or 1
    omp_threads = max(1, cpu_threads // max(1, args.nproc_per_node))
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    cmd = _build_command(
        repo_root=repo_root,
        dataset_dir=dataset_dir,
        checkpoint=checkpoint,
        predictions=predictions,
        top_k=args.top_k,
        nproc_per_node=args.nproc_per_node,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print("$ " + shlex.join(cmd))

    if args.dry_run:
        return 0

    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
