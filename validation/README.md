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
