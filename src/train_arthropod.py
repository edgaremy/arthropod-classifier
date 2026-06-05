from __future__ import annotations

import argparse
import importlib.util
import shlex
import sys
from pathlib import Path


DEFAULT_NPROC_PER_NODE = 2
DEFAULT_DATASET_DIR = 'dataset'
DEFAULT_OUTPUT_DIR = 'output/arthropod-classifier'


def _count_classes(class_map_path: Path) -> int:
    with class_map_path.open('r', encoding='utf-8') as handle:
        return sum(1 for line in handle if line.strip())


def _load_timm_train_module(repo_root: Path):
    timm_repo = repo_root / 'pytorch-image-models'
    train_script = timm_repo / 'train.py'

    if not train_script.exists():
        raise FileNotFoundError(f'timm train.py not found at: {train_script}')

    if str(timm_repo) not in sys.path:
        sys.path.insert(0, str(timm_repo))

    spec = importlib.util.spec_from_file_location('arthropod_timm_train', train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load module spec for: {train_script}')

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_modified_timm_package(repo_root: Path):
    package_dir = repo_root / 'src' / 'modified_timm'
    init_file = package_dir / '__init__.py'

    if not init_file.exists():
        raise FileNotFoundError(f'modified_timm package not found at: {init_file}')

    spec = importlib.util.spec_from_file_location(
        'modified_timm',
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load module spec for: {init_file}')

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_command(repo_root: Path, dataset_dir: Path, output_dir: Path, nproc_per_node: int) -> list[str]:
    train_entry = repo_root / 'src' / 'train_arthropod.py'
    class_map = dataset_dir / 'class-mapping.txt'

    if not train_entry.exists():
        raise FileNotFoundError(f'Missing train entrypoint: {train_entry}')
    if not class_map.exists():
        raise FileNotFoundError(f'Missing class map file: {class_map}')

    num_classes = _count_classes(class_map)
    if num_classes <= 0:
        raise RuntimeError(f'No classes found in class map: {class_map}')

    return [
        'torchrun',
        '--nproc_per_node',
        str(nproc_per_node),
        str(train_entry),
        '--data-dir',
        str(dataset_dir),
        '--model',
        'convnextv2_base.fcmae_ft_in22k_in1k_384',
        '--pretrained',
        '--num-classes',
        str(num_classes),
        '--input-size',
        '3',
        '384',
        '384',
        '--class-map',
        str(class_map),
        '--epochs',
        '100',
        '-b',
        '32',
        '-vb',
        '64',
        '-j',
        '16',
        '--log-interval',
        '200',
        '--opt',
        'lamb',
        '--lr',
        '3e-4',
        '--sched',
        'cosine',
        '--weight-decay',
        '0.01',
        '--warmup-epochs',
        '5',
        '--smoothing',
        '0.1',
        '--drop-path',
        '0.05',
        '--mixup',
        '0.2',
        '--cutmix',
        '1.0',
        '--hflip',
        '0.5',
        '--aa',
        'rand-m7-mstd0.5',
        '--bce-loss',
        '--amp',
        '--eval-metric',
        'f1_macro',
        '--output',
        str(output_dir),
    ]


def _parse_wrapper_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--arthropod-list-tweaks',
        action='store_true',
        help='List registered arthropod tweaks and exit.',
    )
    parser.add_argument(
        '--arthropod-disable-tweak',
        action='append',
        default=[],
        metavar='NAME',
        help='Disable a tweak by name. Can be provided multiple times.',
    )
    parser.add_argument(
        '--arthropod-no-strict',
        action='store_true',
        help='Do not fail if tweak compatibility anchors are missing.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print command without running it.',
    )
    return parser.parse_known_args(argv)


def main() -> int:
    wrapper_args, passthrough = _parse_wrapper_args(sys.argv[1:])

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    dataset_dir = (repo_root / DEFAULT_DATASET_DIR).resolve()
    output_dir = (repo_root / DEFAULT_OUTPUT_DIR).resolve()
    cmd = _build_command(
        repo_root=repo_root,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        nproc_per_node=DEFAULT_NPROC_PER_NODE,
    )
    print('$ ' + shlex.join(cmd))

    if wrapper_args.dry_run:
        return 0

    train_module = _load_timm_train_module(repo_root)
    modified_timm = _load_modified_timm_package(repo_root)
    registry = modified_timm.build_default_registry()

    if wrapper_args.arthropod_list_tweaks:
        print('Registered arthropod timm tweaks:')
        for name, description in registry.describe():
            print(f' - {name}: {description}')
        return 0

    results = registry.apply(
        train_module,
        disabled=set(wrapper_args.arthropod_disable_tweak),
        strict=not wrapper_args.arthropod_no_strict,
    )
    for result in results:
        state = 'applied' if result.applied else 'skipped'
        print(f'[{state}] {result.name}: {result.reason}')

    sys.argv = [sys.argv[0], *passthrough]
    train_module.main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
