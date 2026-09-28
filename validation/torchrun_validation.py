#!/usr/bin/env python3
"""
Validate the best arthropod-classifier checkpoint with timm's validate.py.

This single file plays the two roles that training splits across
training/torchrun_arthropod.py (launcher) and src/train_arthropod.py (entry):

1. Launcher mode (default, no --entry): builds a torchrun command that
   re-invokes this same file in entry mode. Use it for local runs:

       python validation/torchrun_validation.py [--dry-run]

2. Entry mode (--entry): loads the vendored timm validate.py
   (pytorch-image-models/validate.py), applies the arthropod validation
   tweaks from src.modified_timm (f1_macro + per-class F1, matching the
   training eval metric), and runs timm validation in-process.

NOTE: timm's validate.py is single-process (multi-GPU would go through
DataParallel's --num-gpu, not torchrun). torchrun is used with
--nproc-per-node 1 for parity with the training launcher; higher values
would duplicate the whole evaluation once per process.

The sbatch equivalent for the cluster ARM GPU nodes is
validation/sbatch_validation.sh, which runs the entry mode via srun+gbind.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path


# Editable defaults.
DEFAULT_NPROC_PER_NODE = 1
DEFAULT_DATASET_DIR = "dataset"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL = "convnextv2_base.fcmae_ft_in22k_in1k_384"
DEFAULT_CHECKPOINT = "output/arthropod-classifier/20260926-234704-convnextv2_base_fcmae_ft_in22k_in1k_384-384/checkpoint-98.pth.tar"
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 16
DEFAULT_INPUT_SIZE = ["3", "384", "384"]


def _count_classes(class_map_path: Path) -> int:
    with class_map_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _resolve_path(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _default_results_file(checkpoint: Path) -> Path:
    # checkpoint-98.pth.tar -> test_checkpoint-98.csv next to the checkpoint.
    stem = checkpoint.name
    for suffix in checkpoint.suffixes:
        stem = stem[: -len(suffix)]
    return checkpoint.parent / f"test_{stem}.csv"


def _build_validation_args(
    repo_root: Path,
    dataset_dir: Path,
    checkpoint: Path,
    results_file: Path,
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

    return [
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
        "--metrics-avg",
        "macro",
        "--results-file",
        str(results_file),
    ]


def _build_command(
    repo_root: Path,
    dataset_dir: Path,
    checkpoint: Path,
    results_file: Path,
    nproc_per_node: int,
    batch_size: int,
    workers: int,
) -> list[str]:
    validation_args = _build_validation_args(
        repo_root=repo_root,
        dataset_dir=dataset_dir,
        checkpoint=checkpoint,
        results_file=results_file,
        batch_size=batch_size,
        workers=workers,
    )
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(nproc_per_node),
        str(Path(__file__).resolve()),
        "--entry",
        *validation_args,
    ]


def _load_timm_validate_module(repo_root: Path):
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


def _load_modified_timm_package(repo_root: Path):
    package_dir = repo_root / "src" / "modified_timm"
    init_file = package_dir / "__init__.py"

    if not init_file.exists():
        raise FileNotFoundError(f"modified_timm package not found at: {init_file}")

    spec = importlib.util.spec_from_file_location(
        "modified_timm",
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for: {init_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_validation_entry(wrapper_args, passthrough: list[str]) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    validate_module = _load_timm_validate_module(repo_root)
    modified_timm = _load_modified_timm_package(repo_root)
    registry = modified_timm.build_validation_registry()

    if wrapper_args.arthropod_list_tweaks:
        print("Registered arthropod timm validation tweaks:")
        for name, description in registry.describe():
            print(f" - {name}: {description}")
        return 0

    results = registry.apply(
        validate_module,
        disabled=set(wrapper_args.arthropod_disable_tweak),
        strict=not wrapper_args.arthropod_no_strict,
    )
    for result in results:
        state = "applied" if result.applied else "skipped"
        print(f"[{state}] {result.name}: {result.reason}")

    sys.argv = [sys.argv[0], *passthrough]
    validate_module.main()
    return 0


def _parse_wrapper_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--entry",
        action="store_true",
        help="Run in entry mode: execute timm validation in-process (used by the launcher and sbatch_validation.sh).",
    )
    parser.add_argument(
        "--arthropod-list-tweaks",
        action="store_true",
        help="List registered arthropod validation tweaks and exit.",
    )
    parser.add_argument(
        "--arthropod-disable-tweak",
        action="append",
        default=[],
        metavar="NAME",
        help="Disable a tweak by name. Can be provided multiple times.",
    )
    parser.add_argument(
        "--arthropod-no-strict",
        action="store_true",
        help="Do not fail if tweak compatibility anchors are missing.",
    )
    return parser.parse_known_args(argv)


def main() -> int:
    wrapper_args, passthrough = _parse_wrapper_args(sys.argv[1:])

    if wrapper_args.entry:
        return _run_validation_entry(wrapper_args, passthrough)

    parser = argparse.ArgumentParser(description="Launch timm validation with torchrun.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Dataset directory containing train/val/test and class-mapping.txt")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to the checkpoint to validate")
    parser.add_argument("--results-file", default="", help="Output CSV file for validation results (default: test_<checkpoint>.csv next to the checkpoint)")
    parser.add_argument("--nproc-per-node", type=int, default=DEFAULT_NPROC_PER_NODE, help="Number of processes per node (timm validate.py is single-process, keep 1)")
    parser.add_argument("-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Validation batch size")
    parser.add_argument("-j", "--workers", type=int, default=DEFAULT_WORKERS, help="Number of data loading workers")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running it")
    args = parser.parse_args(passthrough)

    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = _resolve_path(repo_root, args.dataset_dir)
    checkpoint = _resolve_path(repo_root, args.checkpoint)
    results_file = (
        _resolve_path(repo_root, args.results_file)
        if args.results_file
        else _default_results_file(checkpoint)
    )
    cpu_threads = os.cpu_count() or 1
    omp_threads = max(1, cpu_threads // max(1, args.nproc_per_node))
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    cmd = _build_command(
        repo_root=repo_root,
        dataset_dir=dataset_dir,
        checkpoint=checkpoint,
        results_file=results_file,
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
