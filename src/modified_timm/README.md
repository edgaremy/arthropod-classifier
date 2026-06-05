# Arthropod timm tweak layer

This directory centralizes runtime tweaks applied to the vendored timm training script.

## Why this exists

We keep `pytorch-image-models/` unmodified and inject controlled changes at runtime from `src/`.
This avoids drift from ad-hoc edits and gives one place to maintain future customizations.

## Entrypoint

Use:

```bash
python src/train_arthropod.py <timm train.py args>
```

Example:

```bash
python src/train_arthropod.py --data-dir dataset --model convnextv2_base.fcmae_ft_in22k_in1k_384 --eval-metric f1_macro
```

## Built-in tweak

- `f1_macro_validation`: patches timm runtime to:
	- append `f1_macro` to validation metrics
	- log validation per-class F1 to `per_class_f1_val.csv`
	- log training per-class F1 to `per_class_f1_train.csv`

This enables:

```bash
--eval-metric f1_macro
```

for best-checkpoint selection.

CSV format:

- Header: `epoch,f1_macro,<class_0>...` (or class names from `--class-map` when length matches)
- Validation epoch index is inferred from existing CSV rows.
- Training epoch index uses timm's `epoch` argument.

## Wrapper controls

- `--arthropod-list-tweaks`: print available tweaks and exit
- `--arthropod-disable-tweak NAME`: disable one tweak (repeatable)
- `--arthropod-no-strict`: skip hard failure when timm source anchors changed

## Adding future tweaks

1. Add a new tweak class in `src/modified_timm/tweaks/`.
2. Implement `.apply(train_module, strict=True)` and return `TweakResult`.
3. Register the tweak in `build_default_registry()` in `src/modified_timm/registry.py`.
4. Keep anchor checks to fail fast on upstream timm changes.
