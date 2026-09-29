# Arthropod Classifier Validation

This directory contains the scripts for evaluating the best arthropod-classifier
checkpoint (`convnextv2_base.fcmae_ft_in22k_in1k_384`,
`output/arthropod-classifier/20260926-234704-convnextv2_base_fcmae_ft_in22k_in1k_384-384/checkpoint-98.pth.tar`)
on the dataset **test split**.

There are two jobs, each with a local launcher and a cluster (SBATCH) variant:

| | Local launcher | Cluster (ARM GPU nodes) |
|---|---|---|
| Metrics validation (Acc@1/5, macro-F1, per-class F1) | `torchrun_validation.py` | `sbatch_validation.sh` |
| Sparse top-k prediction saving | `torchrun_save_predictions.py` | `sbatch_save_predictions.sh` |

Post-processing of the saved predictions (CPU only, x86 env, no GPU/ARM):

| | Local | Cluster (CPU partition) |
|---|---|---|
| F1-macro vs confidence threshold | `metrics/F1-macro_threshold.py` | `metrics/sbatch_F1-macro_threshold.sh` |
| Taxonomy-level metrics (species → class) | `metrics/taxonomy_metrics.py` | `metrics/sbatch_taxonomy_metrics.sh` |

Both Python scripts follow the training layout in one file: by default they are
**launcher mode** wrappers that build and run a `torchrun --nproc-per-node 1`
command re-invoking themselves with `--entry`; in **entry mode** they load the
vendored timm (`pytorch-image-models/`) and run in-process. timm's validation
path is single-process, so `--nproc-per-node` should stay at 1.

## Prerequisites

- Dataset at `/work/datasets/arthropod/` with `train/`, `val/`, `test/` and
  `class-mapping.txt` (24,206 classes). Locally the repo's `dataset/` directory
  resolves to the same data.
- On the cluster GPU nodes (`kairosgh*`, ARM/GH200): the user-owned ARM conda
  env at `/users/p26060/p26060rmdg/arm-envs/arthropod-classifier` (created once
  with `bash training/setup_arm_env.sh`). The x86 conda env cannot run there.

## `torchrun_validation.py`

Metrics validation using timm's `validate.py`. Computes top-1/top-5 accuracy
plus macro precision/recall/F1 (`--metrics-avg macro`), and applies the
`src.modified_timm` validation tweak so the results always include the
training's selection metric `f1_macro` and per-class F1 columns (same metric
pipeline as training).

```bash
python validation/torchrun_validation.py --dry-run   # print the command
python validation/torchrun_validation.py            # run it
```

Options: `--dataset-dir`, `--checkpoint`, `--results-file` (default
`test_<checkpoint>.csv` next to the checkpoint), `-b`, `-j`, `--nproc-per-node`,
`--dry-run`, and the `--arthropod-*` tweak flags.

## `sbatch_validation.sh`

Cluster equivalent of the above: 1 node / 1 task / 1 GPU (`gbind`-bound),
6 h limit, ARM env and module setup mirroring `training/sbatch_train.sh`.
Writes a log to `output/arthropod-classifier/logs/validation_<jobid>_<ts>.log`
and the results CSV next to the checkpoint.

```bash
cd /users/p26060/p26060rmdg/CODE/arthropod-classifier
sbatch validation/sbatch_validation.sh
```

## `torchrun_save_predictions.py`

Saves the model's predictions once so later analyses never re-run the model.
For each image of the test split it stores the **top-k (default k=1000) class
probabilities** as a sparse row, plus the ground-truth label and the image
path. A dense 24,206-probability matrix would be ~104 GB; the top-1000 tail is
numerically irrelevant and fits in a few GB.

The entry mode reuses timm `validate.py`'s argument parser and
model/dataset/loader setup, so preprocessing is identical to the metrics run.
Output path is controlled by the `DEFAULT_PREDICTIONS` string at the top of the
script or the `--predictions` argument; by default predictions go to
`validation/predictions/`.

```bash
python validation/torchrun_save_predictions.py --dry-run   # print the command
python validation/torchrun_save_predictions.py             # run it
```

### Predictions file format

One compressed `.npz` plus a `.json` metadata sidecar with the same stem:

| Key | Shape | Dtype | Content |
|---|---|---|---|
| `values` | (N, k) | float16 | top-k probabilities per image, sorted descending |
| `indices` | (N, k) | uint16 | class indices of `values` (int32 if > 65,535 classes) |
| `labels` | (N,) | uint16 | ground-truth class index per image |
| `files` | (N,) | unicode | image path relative to the split directory |

Row `i` of every array refers to the same image: `files[i]` is its path,
`labels[i]` its true class, `indices[i]`/`values[i]` its top-k probabilities.
Rows follow the sequential eval dataloader order (row `i` is
`dataset.filenames()[i]`), which the script asserts before saving.

```python
import numpy as np

data = np.load("test_checkpoint-98_top1000_predictions.npz")
values, indices, labels, files = (
    data["values"], data["indices"], data["labels"], data["files"]
)

top1_acc = (indices[:, 0] == labels).mean()
mask = np.isin(labels, [123, 4567])        # group of classes of interest
group_top1 = (indices[mask, 0] == labels[mask]).mean()

# top-5 check without re-running the model
top5_acc = (indices[:, :5] == labels[:, None]).any(axis=1).mean()
```

Notes:

- Probabilities are the softmax over all classes; only the top-k entries are
  kept, so rows do not sum exactly to 1 (they do so up to the dropped tail).
- For an image whose true class is outside its own top-k, the class's
  probability is not recoverable from the file; for k=1000 this is a corner
  case (badly misclassified images only).
- The `.json` sidecar records the checkpoint, dataset, split, `k`, dtypes and
  row-order guarantee.

## `sbatch_save_predictions.sh`

Cluster equivalent of the prediction saving: same node/task/GPU layout as
`sbatch_validation.sh`, writes a log to
`output/arthropod-classifier/logs/predictions_<jobid>_<ts>.log` and the
`.npz` to `validation/predictions/`. It refuses to run if the predictions
file already exists, to protect the artifact from accidental overwrites.

```bash
cd /users/p26060/p26060rmdg/CODE/arthropod-classifier
sbatch validation/sbatch_save_predictions.sh
```

## Metrics from saved predictions

### `metrics/F1-macro_threshold.py`

Sweeps a minimum confidence threshold from 0 (no threshold) to 1 and plots
F1-macro against it, using only the saved top-k predictions (no model run, no
GPU). A top-1 prediction whose confidence is below the threshold is not
counted: it contributes no TP and no FP for the predicted class (a
low-confidence wrong prediction no longer penalizes that class' precision);
under the main curve the rejection counts as a miss (FN) for the true class.
The plot also shows a "covered only" variant (rejected images excluded
entirely) and the coverage (fraction of predictions still counted). F1-macro
is computed over all 24,206 classes with zero_division=0, matching the
training's `f1_macro`.

```bash
python validation/metrics/F1-macro_threshold.py \
    [--predictions validation/predictions/test_checkpoint-98_top1000_predictions.npz] \
    [--step 0.005]
```

Outputs, under `validation/metrics/F1-macro/`:

- `plots/<predictions>_vs_threshold.png` - the plot, with the best threshold
  annotated
- `<predictions>_threshold_sweep.csv` - threshold, f1_macro,
  f1_macro_covered, coverage per point

### `metrics/sbatch_F1-macro_threshold.sh`

Cluster equivalent on the CPU partition: MicroShared CPU (`micro-cpu`, 16
task slots, non-exclusive), x86 conda env, single `srun -n 1` process. The
repo and the predictions file already live on GPFS (`$WORK`). It fails early
if the predictions file is missing.

```bash
cd /users/p26060/p26060rmdg/CODE/arthropod-classifier
sbatch validation/metrics/sbatch_F1-macro_threshold.sh
```

### `metrics/taxonomy_metrics.py`

Computes metrics at five taxonomic scales - species, genus, family, order,
class - by grouping the stored species probabilities per taxon (the taxonomy
comes from the dataset's `class-names.csv`, joined on `class-mapping.txt`).
For each level: top-1/top-5 accuracy, and per-group precision/recall/F1
(one-vs-rest, from the top-1 group prediction); the -macro versions average
over groups (zero_division=0, matching the training's `f1_macro`).

```bash
python validation/metrics/taxonomy_metrics.py \
    [--predictions validation/predictions/test_checkpoint-98_top1000_predictions.npz] \
    [--class-names dataset/class-names.csv] [--class-mapping dataset/class-mapping.txt] \
    [--levels species,genus,family,order,class] [--ignore-ungrouped] [--no-renormalize]
```

Behavior details:

- Group scores are the summed species probabilities, renormalized per row:
  the stored top-1000 probabilities are a truncated softmax, so the raw
  sums miss the tail mass. Per-row renormalization does not change
  top-1/top-5 decisions (positive scaling preserves ranking); disable with
  `--no-renormalize`.
- Species missing a taxon (e.g. an empty `order`) form a synthetic
  "(no `<level>`)" group by default. With `--ignore-ungrouped`, images
  whose true species has no group are excluded from that level (a top-1
  prediction falling into the dropped group can then be neither TP nor FP);
  the number of images without a group and the number ignored are reported
  per level in `summary.csv`.

Outputs, under `validation/metrics/taxonomy/`:

```
taxonomy/
├── group_mapping.csv      # shared: species index -> taxon names and ids (the
│                          #   aggregation table actually used)
└── <model>/               # e.g. convnextv2_base (from the predictions
    │                      #   metadata; one directory per model)
    ├── summary.csv        # one row per level: n_groups, top1_acc, top5_acc,
    │                      #   precision_macro, recall_macro, f1_macro,
    │                      #   f1_macro_supported, n_images_without_group,
    │                      #   n_ignored, ...
    ├── per_group/<level>.csv   # one row per group: n_species, n_test_images,
    │                           #   tp/fp/fn, precision, recall, f1, top1, top5
    └── <predictions stem>_taxonomy_metrics.json   # metadata (input files +
                                                    #   sha256, options used)
```

`f1_macro` averages over all groups at the level (zero-division groups
count as 0, consistent with the species-level definition);
`f1_macro_supported` averages only groups with test images.

### `metrics/sbatch_taxonomy_metrics.sh`

Cluster equivalent on the CPU partition: MicroShared CPU (`micro-cpu`, 16
task slots, non-exclusive), x86 conda env, single `srun -n 1` process
(about 15-25 minutes for ~1M images over all five levels). It fails early
if the predictions file or `class-names.csv` is missing. Taxonomy options
(renormalization, `--ignore-ungrouped`) can be set via the `TAXONOMY_ARGS`
variable inside the script.

```bash
cd /users/p26060/p26060rmdg/CODE/arthropod-classifier
sbatch validation/metrics/sbatch_taxonomy_metrics.sh
```
