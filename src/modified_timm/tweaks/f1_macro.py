from __future__ import annotations

import csv
import functools
import inspect
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

from ..models import TweakResult


@dataclass(frozen=True)
class MacroF1ValidationTweak:
    name: str = 'f1_macro_validation'
    description: str = (
        'Add f1_macro metric support and log per-class F1 CSVs for validation and training.'
    )

    _validate_anchors: tuple[str, ...] = (
        'acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))',
        "metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])",
    )
    _train_anchors: tuple[str, ...] = (
        'result = task(input, target)',
        "return OrderedDict([('loss', loss_avg)])",
    )

    def apply(self, train_module: types.ModuleType, strict: bool = True) -> TweakResult:
        validate_fn = getattr(train_module, 'validate', None)
        train_one_epoch_fn = getattr(train_module, 'train_one_epoch', None)
        if validate_fn is None:
            msg = 'train module has no validate() function'
            if strict:
                raise RuntimeError(msg)
            return TweakResult(name=self.name, applied=False, reason=msg)
        if train_one_epoch_fn is None:
            msg = 'train module has no train_one_epoch() function'
            if strict:
                raise RuntimeError(msg)
            return TweakResult(name=self.name, applied=False, reason=msg)

        validate_src = inspect.getsource(validate_fn)
        missing_validate = [anchor for anchor in self._validate_anchors if anchor not in validate_src]
        train_src = inspect.getsource(train_one_epoch_fn)
        missing_train = [anchor for anchor in self._train_anchors if anchor not in train_src]
        if missing_validate or missing_train:
            msg = (
                'train module signature changed, missing anchors: '
                f'validate={missing_validate}, train_one_epoch={missing_train}'
            )
            if strict:
                raise RuntimeError(msg)
            return TweakResult(name=self.name, applied=False, reason=msg)

        self._patch_get_outdir(train_module)
        train_module.validate = self._build_validate(train_module, validate_fn)
        train_module.train_one_epoch = self._build_train_one_epoch(train_module, train_one_epoch_fn)
        return TweakResult(name=self.name, applied=True, reason='ok')

    @staticmethod
    def _import_f1_score():
        try:
            from sklearn.metrics import f1_score as _sk_f1
        except ImportError as exc:
            raise RuntimeError(
                'Macro F1 tweak requires scikit-learn. Install it with: pip install scikit-learn'
            ) from exc
        return _sk_f1

    @staticmethod
    def _patch_get_outdir(train_module: types.ModuleType) -> None:
        if getattr(train_module, '_arthropod_outdir_capture_installed', False):
            return

        original_get_outdir = train_module.utils.get_outdir

        @functools.wraps(original_get_outdir)
        def _capturing_get_outdir(*args, **kwargs):
            out_dir = original_get_outdir(*args, **kwargs)
            train_module._arthropod_last_output_dir = out_dir
            return out_dir

        train_module.utils.get_outdir = _capturing_get_outdir
        train_module._arthropod_outdir_capture_installed = True

    @staticmethod
    def _resolve_output_dir(train_module: types.ModuleType, args) -> Path:
        out_dir = getattr(train_module, '_arthropod_last_output_dir', None)
        if not out_dir:
            out_dir = getattr(args, 'output', '') or '.'
        return Path(out_dir)

    @staticmethod
    def _class_names_from_args(args, n_classes: int) -> list[str]:
        default_names = [f'class_{i}' for i in range(n_classes)]
        class_map_path = getattr(args, 'class_map', None)
        if not class_map_path:
            return default_names

        try:
            names = []
            with open(class_map_path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        names.append(stripped)
            if len(names) != n_classes:
                return default_names
            return names
        except Exception:
            return default_names

    @staticmethod
    def _infer_epoch_from_csv(csv_path: Path) -> int:
        if not csv_path.exists():
            return 0
        try:
            with csv_path.open('r', newline='', encoding='utf-8') as handle:
                # Header + N rows means current epoch index N.
                return max(0, sum(1 for _ in handle) - 1)
        except Exception:
            return 0

    @classmethod
    def _write_per_class_csv(
        cls,
        train_module: types.ModuleType,
        args,
        filename: str,
        f1_macro: float,
        f1_per_class: list[float],
        epoch: int | None,
    ) -> None:
        if not train_module.utils.is_primary(args):
            return

        out_dir = cls._resolve_output_dir(train_module, args)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / filename
        class_names = cls._class_names_from_args(args, len(f1_per_class))
        write_header = not csv_path.exists()
        epoch_idx = cls._infer_epoch_from_csv(csv_path) if epoch is None else int(epoch)

        try:
            with csv_path.open('a', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                if write_header:
                    writer.writerow(['epoch', 'f1_macro'] + class_names)
                writer.writerow([epoch_idx, round(float(f1_macro), 6)] + [round(float(v), 6) for v in f1_per_class])
        except Exception:
            # Logging should never crash training.
            return

    @staticmethod
    def _gather_lists_distributed(args, preds: list[int], targets: list[int]) -> tuple[list[int], list[int]]:
        if not getattr(args, 'distributed', False):
            return preds, targets
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return preds, targets

        gathered_preds = [None] * args.world_size
        gathered_targets = [None] * args.world_size
        torch.distributed.all_gather_object(gathered_preds, preds)
        torch.distributed.all_gather_object(gathered_targets, targets)

        flat_preds = [pred for rank_preds in gathered_preds for pred in rank_preds]
        flat_targets = [label for rank_targets in gathered_targets for label in rank_targets]
        return flat_preds, flat_targets

    @classmethod
    def _build_validate(cls, train_module: types.ModuleType, original_validate):
        _sk_f1 = cls._import_f1_score()

        @functools.wraps(original_validate)
        def _validate_with_f1(
            model,
            loader,
            loss_fn,
            args,
            device=torch.device('cuda'),
            amp_autocast=train_module.suppress,
            model_dtype=None,
            log_suffix='',
        ):
            all_preds: list[int] = []
            all_targets: list[int] = []

            original_accuracy = train_module.utils.accuracy

            @functools.wraps(original_accuracy)
            def _capturing_accuracy(output, target, topk=(1,)):
                batch_preds = output.argmax(1).detach()
                batch_targets = target.detach() if target.ndim == 1 else target.argmax(1).detach()
                all_preds.extend(batch_preds.cpu().tolist())
                all_targets.extend(batch_targets.cpu().tolist())
                return original_accuracy(output, target, topk=topk)

            train_module.utils.accuracy = _capturing_accuracy
            try:
                metrics = original_validate(
                    model,
                    loader,
                    loss_fn,
                    args,
                    device=device,
                    amp_autocast=amp_autocast,
                    model_dtype=model_dtype,
                    log_suffix=log_suffix,
                )
            finally:
                train_module.utils.accuracy = original_accuracy

            all_preds, all_targets = cls._gather_lists_distributed(args, all_preds, all_targets)
            if all_preds and all_targets:
                f1_macro = float(_sk_f1(all_targets, all_preds, average='macro', zero_division=0))
                f1_per_class = _sk_f1(all_targets, all_preds, average=None, zero_division=0).tolist()
                metrics = OrderedDict(metrics)
                metrics['f1_macro'] = f1_macro
                cls._write_per_class_csv(
                    train_module=train_module,
                    args=args,
                    filename='per_class_f1_val.csv',
                    f1_macro=f1_macro,
                    f1_per_class=f1_per_class,
                    epoch=None,
                )

            return metrics

        return _validate_with_f1

    @classmethod
    def _build_train_one_epoch(cls, train_module: types.ModuleType, original_train_one_epoch):
        _sk_f1 = cls._import_f1_score()

        @functools.wraps(original_train_one_epoch)
        def _train_one_epoch_with_f1(
            epoch,
            model,
            loader,
            optimizer,
            args,
            task=None,
            device=torch.device('cuda'),
            lr_scheduler=None,
            saver=None,
            output_dir=None,
            amp_autocast=train_module.suppress,
            loss_scaler=None,
            model_dtype=None,
            mixup_fn=None,
            num_updates_total=None,
            naflex_mode=False,
            scheduled_batch_mode=False,
            batch_size_reference=None,
        ):
            all_preds: list[int] = []
            all_targets: list[int] = []

            if task is None:
                return original_train_one_epoch(
                    epoch,
                    model,
                    loader,
                    optimizer,
                    args,
                    task=task,
                    device=device,
                    lr_scheduler=lr_scheduler,
                    saver=saver,
                    output_dir=output_dir,
                    amp_autocast=amp_autocast,
                    loss_scaler=loss_scaler,
                    model_dtype=model_dtype,
                    mixup_fn=mixup_fn,
                    num_updates_total=num_updates_total,
                    naflex_mode=naflex_mode,
                    scheduled_batch_mode=scheduled_batch_mode,
                    batch_size_reference=batch_size_reference,
                )

            class _TaskCaptureProxy:
                def __init__(self, inner_task):
                    self._inner_task = inner_task

                def __call__(self, input_batch, target_batch):
                    result = self._inner_task(input_batch, target_batch)
                    output = result.get('output') if isinstance(result, dict) else None
                    if torch.is_tensor(output) and output.ndim >= 2:
                        preds = output.detach().argmax(1)
                        tgt_hard = target_batch if target_batch.ndim == 1 else target_batch.argmax(1)
                        all_preds.extend(preds.cpu().tolist())
                        all_targets.extend(tgt_hard.detach().cpu().tolist())
                    return result

                def __getattr__(self, item):
                    return getattr(self._inner_task, item)

            metrics = original_train_one_epoch(
                epoch,
                model,
                loader,
                optimizer,
                args,
                task=_TaskCaptureProxy(task),
                device=device,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_dtype=model_dtype,
                mixup_fn=mixup_fn,
                num_updates_total=num_updates_total,
                naflex_mode=naflex_mode,
                scheduled_batch_mode=scheduled_batch_mode,
                batch_size_reference=batch_size_reference,
            )

            all_preds, all_targets = cls._gather_lists_distributed(args, all_preds, all_targets)
            if all_preds and all_targets:
                f1_train_macro = float(_sk_f1(all_targets, all_preds, average='macro', zero_division=0))
                f1_train_per_class = _sk_f1(all_targets, all_preds, average=None, zero_division=0).tolist()
                cls._write_per_class_csv(
                    train_module=train_module,
                    args=args,
                    filename='per_class_f1_train.csv',
                    f1_macro=f1_train_macro,
                    f1_per_class=f1_train_per_class,
                    epoch=epoch,
                )

            return metrics

        return _train_one_epoch_with_f1
