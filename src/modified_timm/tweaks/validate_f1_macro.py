from __future__ import annotations

import functools
import inspect
import re
import types
from collections import OrderedDict
from dataclasses import dataclass

import torch

from ..models import TweakResult


@dataclass(frozen=True)
class MacroF1ValidateScriptTweak:
    name: str = "f1_macro_validate_script"
    description: str = (
        "Patch timm validate.py to always compute f1_macro and append per-class F1 columns to results."
    )

    _validate_anchors: tuple[str, ...] = (
        "acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))",
        "results = OrderedDict(",
    )

    def apply(self, train_module: types.ModuleType, strict: bool = True) -> TweakResult:
        validate_fn = getattr(train_module, "validate", None)
        if validate_fn is None:
            msg = "module has no validate() function"
            if strict:
                raise RuntimeError(msg)
            return TweakResult(name=self.name, applied=False, reason=msg)

        validate_src = inspect.getsource(validate_fn)
        missing_validate = [anchor for anchor in self._validate_anchors if anchor not in validate_src]
        if missing_validate:
            msg = f"validate module signature changed, missing anchors: {missing_validate}"
            if strict:
                raise RuntimeError(msg)
            return TweakResult(name=self.name, applied=False, reason=msg)

        train_module.validate = self._build_validate(train_module, validate_fn)
        return TweakResult(name=self.name, applied=True, reason="ok")

    @staticmethod
    def _import_f1_score():
        try:
            from sklearn.metrics import f1_score as _sk_f1
        except ImportError as exc:
            raise RuntimeError(
                "Macro F1 tweak requires scikit-learn. Install it with: pip install scikit-learn"
            ) from exc
        return _sk_f1

    @staticmethod
    def _class_names_from_args(args, n_classes: int) -> list[str]:
        default_names = [f"class_{i}" for i in range(n_classes)]
        class_map_path = getattr(args, "class_map", None)
        if not class_map_path:
            return default_names

        try:
            names = []
            with open(class_map_path, "r", encoding="utf-8") as handle:
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
    def _sanitize_field_name(name: str) -> str:
        key = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
        key = re.sub(r"_+", "_", key).strip("_")
        return key or "class"

    @classmethod
    def _append_per_class_f1(cls, results: OrderedDict, args, per_class_f1: list[float]) -> None:
        class_names = cls._class_names_from_args(args, len(per_class_f1))
        used: set[str] = set(results.keys())

        for idx, value in enumerate(per_class_f1):
            base = f"f1_{cls._sanitize_field_name(class_names[idx])}"
            key = base
            suffix = 2
            while key in used:
                key = f"{base}_{suffix}"
                suffix += 1
            results[key] = round(100 * float(value), 4)
            used.add(key)

    @classmethod
    def _build_validate(cls, train_module: types.ModuleType, original_validate):
        _sk_f1 = cls._import_f1_score()

        @functools.wraps(original_validate)
        def _validate_with_f1(args):
            all_preds: list[int] = []
            all_targets: list[int] = []

            original_accuracy = train_module.accuracy

            @functools.wraps(original_accuracy)
            def _capturing_accuracy(output, target, topk=(1,)):
                batch_preds = output.argmax(1).detach()
                batch_targets = target.detach() if target.ndim == 1 else target.argmax(1).detach()
                all_preds.extend(batch_preds.cpu().tolist())
                all_targets.extend(batch_targets.cpu().tolist())
                return original_accuracy(output, target, topk=topk)

            train_module.accuracy = _capturing_accuracy
            try:
                results = original_validate(args)
            finally:
                train_module.accuracy = original_accuracy

            if not all_preds or not all_targets:
                return results

            n_classes = max(max(all_preds), max(all_targets)) + 1
            labels = list(range(n_classes))
            f1_macro = float(_sk_f1(all_targets, all_preds, average="macro", zero_division=0, labels=labels))
            f1_per_class = _sk_f1(all_targets, all_preds, average=None, zero_division=0, labels=labels).tolist()

            results = OrderedDict(results)
            results["f1_macro"] = round(100 * f1_macro, 4)
            cls._append_per_class_f1(results, args, f1_per_class)
            return results

        return _validate_with_f1