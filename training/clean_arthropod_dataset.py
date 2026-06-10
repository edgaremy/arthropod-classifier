#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFile, UnidentifiedImageError
from PIL.Image import DecompressionBombError


DEFAULT_DATASET_DIR = 'dataset'
DEFAULT_OUTPUT_FORMAT = 'jpeg'
DEFAULT_MAX_PIXELS = 40_000_000
DEFAULT_MAX_SIDE = 4096
IMAGE_EXTENSIONS = {
    '.bmp',
    '.gif',
    '.jpeg',
    '.jpg',
    '.jpe',
    '.jp2',
    '.png',
    '.ppm',
    '.pgm',
    '.pbm',
    '.pnm',
    '.tif',
    '.tiff',
    '.webp',
}


@dataclass
class ScanStats:
    scanned: int = 0
    clean: int = 0
    resized: int = 0
    deleted: int = 0
    failed: int = 0


def _iter_image_files(dataset_dir: Path, splits: Iterable[str]) -> Iterable[Path]:
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for path in split_dir.rglob('*'):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def _count_image_files(dataset_dir: Path, splits: Iterable[str]) -> int:
    return sum(1 for _ in _iter_image_files(dataset_dir, splits))


def _safe_output_path(path: Path, output_format: str) -> Path:
    suffix = '.jpg' if output_format == 'jpeg' else f'.{output_format}'
    if path.suffix.lower() == suffix:
        return path
    return path.with_suffix(suffix)


def _composite_on_background(image: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        rgba = image.convert('RGBA')
        base = Image.new('RGB', rgba.size, background_color)
        base.paste(rgba, mask=rgba.getchannel('A'))
        return base
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def _resize_image(image: Image.Image, max_pixels: int, max_side: int) -> Image.Image:
    width, height = image.size
    pixel_count = width * height
    if pixel_count <= max_pixels and max(width, height) <= max_side:
        return image

    scale_by_pixels = math.sqrt(max_pixels / pixel_count) if pixel_count > max_pixels else 1.0
    scale_by_side = max_side / max(width, height) if max(width, height) > max_side else 1.0
    scale = min(scale_by_pixels, scale_by_side)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _save_repaired_image(
    source_path: Path,
    image: Image.Image,
    *,
    output_format: str,
    jpeg_quality: int,
    png_compress_level: int,
) -> Path:
    output_path = _safe_output_path(source_path, output_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == 'jpeg':
        save_kwargs: dict[str, object] = {'format': 'JPEG', 'quality': jpeg_quality, 'optimize': True, 'progressive': True}
        image_to_save = _composite_on_background(image, (255, 255, 255))
    elif output_format == 'png':
        save_kwargs = {'format': 'PNG', 'optimize': True, 'compress_level': png_compress_level}
        image_to_save = image
    else:
        save_kwargs = {'format': output_format.upper()}
        image_to_save = image

    tmp_fd, tmp_name = tempfile.mkstemp(prefix=source_path.stem + '.', suffix='.tmp', dir=str(output_path.parent))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        image_to_save.save(tmp_path, **save_kwargs)
        os.replace(tmp_path, output_path)
        source_path.unlink(missing_ok=True)
        return output_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _inspect_image(path: Path) -> tuple[str, tuple[int, int] | None, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return 'ok', image.size, image.mode
    except DecompressionBombError as exc:
        return 'bomb', None, str(exc)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return 'bad', None, str(exc)


def _repair_bombed_image(
    path: Path,
    *,
    max_pixels: int,
    max_side: int,
    output_format: str,
    jpeg_quality: int,
    png_compress_level: int,
) -> Path:
    previous_limit = Image.MAX_IMAGE_PIXELS
    previous_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(path) as image:
            image.load()
            resized = _resize_image(image, max_pixels=max_pixels, max_side=max_side)
            return _save_repaired_image(
                path,
                resized,
                output_format=output_format,
                jpeg_quality=jpeg_quality,
                png_compress_level=png_compress_level,
            )
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated


def _delete_path(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.unlink(missing_ok=True)


def _format_progress(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return '[processing]'

    ratio = min(1.0, current / total)
    filled = min(width, int(round(width * ratio)))
    bar = '#' * filled + '-' * (width - filled)
    percent = int(round(100 * ratio))
    return f'[{bar}] {current}/{total} ({percent:3d}%)'


def _print_progress(current: int, total: int) -> None:
    print(f'\r{_format_progress(current, total)}', end='', file=sys.stderr, flush=True)


def _finish_progress(total: int) -> None:
    if total > 0:
        print(f'\r{_format_progress(total, total)}', file=sys.stderr, flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Scan and clean arthropod dataset images.')
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR, help='Dataset root containing train/val/test folders.')
    parser.add_argument('--splits', nargs='*', default=['train', 'val', 'test'], help='Dataset splits to scan.')
    parser.add_argument('--max-pixels', type=int, default=DEFAULT_MAX_PIXELS, help='Maximum allowed pixel count after repair.')
    parser.add_argument('--max-side', type=int, default=DEFAULT_MAX_SIDE, help='Maximum allowed side length after repair.')
    parser.add_argument('--output-format', choices=['jpeg', 'png'], default=DEFAULT_OUTPUT_FORMAT, help='Format used when rewriting repaired images.')
    parser.add_argument('--jpeg-quality', type=int, default=88, help='JPEG quality used when saving repaired images.')
    parser.add_argument('--png-compress-level', type=int, default=9, help='PNG compression level used when saving repaired images.')
    parser.add_argument('--unreadable-action', choices=['delete'], default='delete', help='Action for images Pillow cannot identify.')
    parser.add_argument('--bomb-action', choices=['resize', 'delete'], default='resize', help='Action for images that exceed Pillow pixel limits.')
    parser.add_argument('--dry-run', action='store_true', help='Report actions without changing any files.')
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = (repo_root / args.dataset_dir).resolve() if not Path(args.dataset_dir).is_absolute() else Path(args.dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f'Dataset directory not found: {dataset_dir}')

    total_files = _count_image_files(dataset_dir, args.splits)
    stats = ScanStats()
    for index, path in enumerate(_iter_image_files(dataset_dir, args.splits), start=1):
        _print_progress(index, total_files)
        stats.scanned += 1
        state, size, detail = _inspect_image(path)

        if state == 'ok':
            stats.clean += 1
            continue

        if state == 'bomb':
            if args.bomb_action == 'delete':
                print(f'[bomb -> delete] {path}')
                _delete_path(path, dry_run=args.dry_run)
                stats.deleted += 1
                continue

            print(f'[bomb -> resize] {path}')
            if not args.dry_run:
                try:
                    repaired_path = _repair_bombed_image(
                        path,
                        max_pixels=args.max_pixels,
                        max_side=args.max_side,
                        output_format=args.output_format,
                        jpeg_quality=args.jpeg_quality,
                        png_compress_level=args.png_compress_level,
                    )
                    print(f'  saved {repaired_path}')
                    stats.resized += 1
                except Exception as exc:
                    print(f'  failed to repair: {exc}')
                    stats.failed += 1
            else:
                stats.resized += 1
            continue

        print(f'[unreadable -> delete] {path} ({detail})')
        _delete_path(path, dry_run=args.dry_run)
        stats.deleted += 1

    _finish_progress(total_files)

    print(
        'Summary: '
        f'scanned={stats.scanned}, '
        f'clean={stats.clean}, '
        f'resized={stats.resized}, '
        f'deleted={stats.deleted}, '
        f'failed={stats.failed}'
    )
    return 0 if stats.failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())