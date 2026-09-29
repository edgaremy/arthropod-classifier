#!/usr/bin/env python3
"""
Taxonomy-level metrics from the saved sparse predictions.

Uses the top-k predictions file produced by
validation/torchrun_save_predictions.py together with the dataset's
class-names.csv (speciesKey -> genus/family/order/class) to evaluate the
model at several taxonomic scales: species, genus, family, order, class.

For each level above species, the stored species probabilities are summed
per group to obtain group scores, and the top-1/top-5 predicted groups are
compared to the true group of the image's ground-truth species. Per-group
precision/recall/F1 (one-vs-rest, from the top-1 group prediction) and
top-1/top-5 accuracy are computed; the -macro versions are the averages
over groups (zero_division=0 for groups without TP/FP/FN, matching the
species-level f1_macro definition used during training).

For every level above species, each per-group CSV row also carries
precision_macro / recall_macro / f1_macro: the mean of the PER-SPECIES
precision/recall/F1 over the species belonging to that group (e.g. a
family row reports the average per-species metric of the species of that
family, a genus row the average over its species, and so on). Only
species with test images in the predictions file are averaged (species
without them would contribute a 0 to every metric and drag the mean
down), which matches the f1_macro computed during training; a group
whose species all lack test images gets 0. The species-level CSV has no
such columns (a species "group" is a single species, so the macro would
just duplicate the row's own metrics).

Outputs (under validation/metrics/taxonomy/):
    group_mapping.csv                       (shared, dataset-only: the
                                            aggregation table actually used)
    <model>/<predictions stem>_taxonomy_metrics.json   (metadata)
    <model>/summary.csv                    (one row per level)
    <model>/per_group/<level>.csv           (one row per group at that level;
                                            genus-to-class levels add the
                                            per-group macro columns above)

<model> is derived from the predictions metadata (e.g. model
"convnextv2_base.fcmae_ft_in22k_in1k_384" -> "convnextv2_base"), so several
models can coexist under taxonomy/ without overwriting each other.

Notes:
- Group scores are renormalized per row: because we only stored the top-1000
  species probabilities (a truncated softmax), the summed group scores are
  missing the tail mass and sum to less than 1. Renormalizing (dividing each
  row by its sum) turns them back into a proper distribution over the groups
  present in the top-k. Per-row scaling does not change the ranking of
  groups, so top-1/top-5 decisions are identical either way; renormalization
  matters if the group scores are used as calibrated probabilities later.
  Disable with --no-renormalize.
- Some species have no entry for a level (e.g. an empty `order` field).
  By default they are assigned to a synthetic group "(no <level>)" which
  appears in the per-group CSV. With --ignore-ungrouped, images whose TRUE
  species has no group are excluded from that level's metrics (counted in
  summary.csv as n_ignored); a top-1 prediction falling into the synthetic
  group can then no longer be a TP nor a FP (the true group gets a FN).

Usage:
    python validation/metrics/taxonomy_metrics.py \
        [--predictions validation/predictions/test_checkpoint-98_top1000_predictions.npz] \
        [--class-names dataset/class-names.csv] [--class-mapping dataset/class-mapping.txt] \
        [--levels species,genus,family,order,class] [--ignore-ungrouped] [--no-renormalize]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


# Editable defaults.
DEFAULT_PREDICTIONS = "validation/predictions/convnextv2_base_top1000_predictions.npz"
DEFAULT_CLASS_NAMES = "dataset/class-names.csv"
DEFAULT_CLASS_MAPPING = "dataset/class-mapping.txt"
DEFAULT_TAXONOMY_DIR = "validation/metrics/taxonomy"
DEFAULT_LEVELS = ("species", "genus", "family", "order", "class")

# Bincount working set: batch_rows * n_groups is capped around this many
# bins (~128 MB of float64 per bincount call).
MAX_BCOUNT_BINS = 16_000_000


def _resolve_path(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_class_mapping(class_mapping_path: Path) -> list[str]:
    with class_mapping_path.open("r", encoding="utf-8") as handle:
        keys = [line.strip() for line in handle if line.strip()]
    if not keys:
        raise RuntimeError(f"No class ids found in: {class_mapping_path}")
    return keys


def _load_taxonomy(class_names_path: Path, class_keys: list[str]) -> dict[str, dict]:
    """Return, for each level, {names: [...], species_to_group: np.ndarray}.

    species_to_group[i] is the group id of class index i; species without a
    group at that level get the synthetic id len(names) (a group named
    "(no <level>)" is appended to names in that case)."""
    with class_names_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {row["speciesKey"]: row for row in rows}

    taxonomy: dict[str, dict] = {}
    levels = [level for level in DEFAULT_LEVELS if level != "species"]
    for level in levels:
        names: list[str] = []
        name_to_id: dict[str, int] = {}
        species_to_group = np.empty(len(class_keys), dtype=np.int32)
        n_missing = 0
        for i, key in enumerate(class_keys):
            row = by_key.get(key)
            group = (row or {}).get(level, "") or ""
            if not group:
                group = f"(no {level})"
                n_missing += 1
            gid = name_to_id.get(group)
            if gid is None:
                gid = len(names)
                name_to_id[group] = gid
                names.append(group)
            species_to_group[i] = gid
        # Keep the synthetic "(no <level>)" group LAST if it exists, so that
        # --ignore-ungrouped can simply drop the last group.
        synthetic = f"(no {level})"
        if synthetic in name_to_id and name_to_id[synthetic] != len(names) - 1:
            old = name_to_id[synthetic]
            last = len(names) - 1
            last_name = names[last]
            names[last], names[old] = synthetic, last_name
            name_to_id[synthetic], name_to_id[last_name] = last, old
            sel = species_to_group == old
            species_to_group[species_to_group == last] = old
            species_to_group[sel] = last
        print(f"  taxonomy '{level}': {len(names)} groups"
              + (f" ({n_missing} species without {level})" if n_missing else ""))
        taxonomy[level] = {"names": names, "species_to_group": species_to_group}
    return taxonomy


def _prf(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    return precision, recall, f1


def _species_level(values: np.ndarray, indices: np.ndarray, labels: np.ndarray, num_classes: int):
    """Metrics with the species themselves as groups (no aggregation)."""
    n = len(labels)
    pred = indices[:, 0].astype(np.int64)
    correct = pred == labels
    wrong = ~correct
    in_top5 = (indices[:, :5] == labels[:, None]).any(axis=1) if indices.shape[1] else np.zeros(n, dtype=bool)

    tp = np.bincount(labels[correct], minlength=num_classes)
    fp = np.bincount(pred[wrong], minlength=num_classes)
    support = np.bincount(labels, minlength=num_classes)
    fn = support - tp
    top1_correct = tp.copy()
    in_top5_per = np.bincount(labels[in_top5], minlength=num_classes)

    return _level_result("species", [str(i) for i in range(num_classes)], tp, fp, fn, support,
                         tp, in_top5_per, np.int64(n), np.int64(0), np.int64(0))


def _grouped_level(level: str, values: np.ndarray, indices: np.ndarray, labels: np.ndarray,
                   group_names: list[str], species_to_group: np.ndarray,
                   ignore_ungrouped: bool, renormalize: bool):
    """Aggregate species probabilities per group in batches and compute the
    level's metrics from the per-row group decisions."""
    n = len(labels)
    g_total = len(group_names)
    synthetic_id = g_total - 1 if group_names[-1].startswith("(no ") else -1
    gid_true_all = species_to_group[labels]

    n_without_group = int((gid_true_all == synthetic_id).sum()) if synthetic_id >= 0 else 0
    n_ignored = n_without_group if ignore_ungrouped else 0

    tp = np.zeros(g_total, dtype=np.int64)
    fp = np.zeros(g_total, dtype=np.int64)
    support = np.zeros(g_total, dtype=np.int64)
    top1_correct = np.zeros(g_total, dtype=np.int64)
    in_top5_per = np.zeros(g_total, dtype=np.int64)
    n_included = 0

    batch = max(1, min(n, MAX_BCOUNT_BINS // g_total))
    k = indices.shape[1]
    top5_k = min(5, g_total)

    for start in range(0, n, batch):
        end = min(start + batch, n)
        b = end - start
        vals = values[start:end].astype(np.float64)
        gids = species_to_group[indices[start:end]]
        gid_true = gid_true_all[start:end]

        included = np.ones(b, dtype=bool)
        if ignore_ungrouped and synthetic_id >= 0:
            included = gid_true != synthetic_id
            if not included.any():
                continue
        gid_true = gid_true[included]
        gids = gids[included]
        vals = vals[included]

        # Sum species probabilities per group: one bincount over flat
        # (row * g_total + group) indices.
        n_rows = len(gid_true)
        flat = np.arange(n_rows)[:, None] * g_total + gids
        gp = np.bincount(flat.ravel(), weights=vals.ravel(), minlength=n_rows * g_total)
        gp = gp.reshape(n_rows, g_total)

        if renormalize:
            # The stored top-k species probabilities are a TRUNCATED softmax:
            # species outside the top-1000 (whose mass is lost) are absent, so
            # the summed group scores are < 1 per row. Renormalizing by the
            # row sum restores a proper distribution over the groups present
            # in the top-k. (A per-row positive scaling does not change the
            # group ranking, so top-1/top-5 decisions are unaffected.)
            row_sums = gp.sum(axis=1, keepdims=True)
            np.divide(gp, row_sums, out=gp, where=row_sums > 0)

        top1_group = gp.argmax(axis=1)
        if top5_k < g_total:
            top5 = np.argpartition(gp, -top5_k, axis=1)[:, -top5_k:]
        else:
            top5 = np.tile(np.arange(g_total), (n_rows, 1))
        hit5 = (top5 == gid_true[:, None]).any(axis=1)

        correct = top1_group == gid_true
        wrong = ~correct
        if ignore_ungrouped and synthetic_id >= 0:
            wrong &= top1_group != synthetic_id  # no FP attributable to a dropped group

        tp += np.bincount(gid_true[correct], minlength=g_total)
        fp += np.bincount(top1_group[wrong], minlength=g_total)
        support += np.bincount(gid_true, minlength=g_total)
        top1_correct += np.bincount(gid_true[correct], minlength=g_total)
        in_top5_per += np.bincount(gid_true[hit5], minlength=g_total)
        n_included += n_rows

    if ignore_ungrouped and synthetic_id >= 0:
        tp, fp = tp[:-1], fp[:-1]
        support, top1_correct, in_top5_per = support[:-1], top1_correct[:-1], in_top5_per[:-1]
        group_names = group_names[:-1]

    fn = support - tp
    return _level_result(level, group_names, tp, fp, fn, support, top1_correct,
                         in_top5_per, np.int64(n_included), np.int64(n_without_group),
                         np.int64(n_ignored))


def _species_macro_per_group(species_metrics: dict, species_to_group: np.ndarray,
                             n_groups: int) -> dict[str, np.ndarray]:
    """Mean of the PER-SPECIES precision/recall/F1 over the species of each
    group, averaging only species with test images (support > 0) so that
    unsupported species do not contribute a 0 to the mean - matching the
    f1_macro computed during training. Groups whose species all lack test
    images get 0.

    Returns {metric: array of length n_groups}; the caller slices off the
    trailing entries for groups dropped from the result (e.g. the synthetic
    group with --ignore-ungrouped, which is kept last)."""
    supported = species_metrics["support"] > 0
    s2g = species_to_group[supported]
    counts = np.bincount(s2g, minlength=n_groups).astype(np.float64)
    macro: dict[str, np.ndarray] = {}
    for metric in ("precision", "recall", "f1"):
        sums = np.bincount(s2g, weights=species_metrics[metric][supported],
                           minlength=n_groups)
        with np.errstate(invalid="ignore", divide="ignore"):
            macro[metric] = np.where(counts > 0, sums / counts, 0.0)
    return macro


def _level_result(level, group_names, tp, fp, fn, support, top1_correct, in_top5_per,
                  n_included, n_without_group, n_ignored):
    precision, recall, f1 = _prf(tp, fp, fn)
    with np.errstate(invalid="ignore", divide="ignore"):
        top1_rate = np.where(support > 0, top1_correct / support, 0.0)
        top5_rate = np.where(support > 0, in_top5_per / support, 0.0)
    return {
        "level": level,
        "group_names": group_names,
        "tp": tp, "fp": fp, "fn": fn, "support": support,
        "precision": precision, "recall": recall, "f1": f1,
        "top1_correct": top1_correct, "in_top5_per": in_top5_per,
        "top1_rate": top1_rate, "top5_rate": top5_rate,
        "n_included": int(n_included),
        "n_without_group": int(n_without_group),
        "n_ignored": int(n_ignored),
    }


def _write_group_csv(path: Path, result: dict, n_species_per_group: np.ndarray,
                     macro: dict[str, np.ndarray] | None = None) -> None:
    """One row per group at the level. For levels above species, `macro` adds
    precision_macro/recall_macro/f1_macro: the mean of the per-species
    metrics over the species of each group (None at the species level)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["group_id", "group_name", "n_species", "n_test_images",
                  "tp", "fp", "fn", "precision", "recall", "f1", "top1", "top5"]
        if macro is not None:
            header += ["precision_macro", "recall_macro", "f1_macro"]
        writer.writerow(header)
        for gid, name in enumerate(result["group_names"]):
            row = [
                gid, name, int(n_species_per_group[gid]), int(result["support"][gid]),
                int(result["tp"][gid]), int(result["fp"][gid]), int(result["fn"][gid]),
                f"{result['precision'][gid]:.6f}", f"{result['recall'][gid]:.6f}",
                f"{result['f1'][gid]:.6f}", f"{result['top1_rate'][gid]:.6f}",
                f"{result['top5_rate'][gid]:.6f}",
            ]
            if macro is not None:
                row += [f"{macro['precision'][gid]:.6f}", f"{macro['recall'][gid]:.6f}",
                        f"{macro['f1'][gid]:.6f}"]
            writer.writerow(row)


def _write_summary(path: Path, results: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["level", "n_groups", "n_groups_with_support", "n_images",
                         "n_images_without_group", "n_ignored",
                         "top1_acc", "top5_acc", "precision_macro", "recall_macro",
                         "f1_macro", "f1_macro_supported"])
        for r in results:
            n_groups = len(r["group_names"])
            has_support = r["support"] > 0
            n_supported = int(has_support.sum())
            included = r["n_included"]
            writer.writerow([
                r["level"], n_groups, n_supported, included,
                r["n_without_group"], r["n_ignored"],
                f"{r['top1_correct'].sum() / included:.6f}" if included else "0.000000",
                f"{r['in_top5_per'].sum() / included:.6f}" if included else "0.000000",
                f"{r['precision'].mean():.6f}",
                f"{r['recall'].mean():.6f}",
                f"{r['f1'].mean():.6f}",
                f"{r['f1'][has_support].mean():.6f}" if n_supported else "0.000000",
            ])


def _write_group_mapping(path: Path, class_keys: list[str], taxonomy: dict) -> None:
    levels = [level for level in DEFAULT_LEVELS if level != "species"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["species_index", "species_key"]
        header += [name for level in levels for name in (level, f"{level}_id")]
        writer.writerow(header)
        for i, key in enumerate(class_keys):
            row = [i, key]
            for level in levels:
                gid = int(taxonomy[level]["species_to_group"][i])
                name = taxonomy[level]["names"][gid]
                row += [name, gid]
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Taxonomy-level metrics (species/genus/family/order/class) from saved top-k predictions.")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS, help="Path to the sparse predictions .npz")
    parser.add_argument("--class-names", default=DEFAULT_CLASS_NAMES, help="Path to class-names.csv (speciesKey -> taxonomy)")
    parser.add_argument("--class-mapping", default="", help="Path to class-mapping.txt (default: from the predictions metadata, else dataset/class-mapping.txt)")
    parser.add_argument("--taxonomy-dir", default=DEFAULT_TAXONOMY_DIR, help="Output directory (per-model subdirectories are created inside)")
    parser.add_argument("--model-name", default="", help="Per-model output subdirectory (default: derived from the predictions metadata model name, e.g. convnextv2_base)")
    parser.add_argument("--levels", default=",".join(DEFAULT_LEVELS), help="Comma-separated levels to compute (default: species,genus,family,order,class)")
    parser.add_argument("--ignore-ungrouped", action="store_true", help="Exclude images whose true species has no group at a level (counted as n_ignored in summary.csv)")
    parser.add_argument("--no-renormalize", action="store_true", help="Keep raw summed group scores instead of renormalizing per row")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    predictions_path = _resolve_path(repo_root, args.predictions)
    class_names_path = _resolve_path(repo_root, args.class_names)
    taxonomy_dir = _resolve_path(repo_root, args.taxonomy_dir)
    levels = [level.strip() for level in args.levels.split(",") if level.strip()]
    unknown = [level for level in levels if level not in DEFAULT_LEVELS]
    if unknown:
        print(f"ERROR: unknown levels: {unknown} (valid: {list(DEFAULT_LEVELS)})", file=sys.stderr)
        return 1

    if not predictions_path.exists():
        print(f"ERROR: predictions file not found: {predictions_path}", file=sys.stderr)
        print("Run validation/sbatch_save_predictions.sh first.", file=sys.stderr)
        return 1
    if not class_names_path.exists():
        print(f"ERROR: class-names.csv not found: {class_names_path}", file=sys.stderr)
        return 1

    metadata_path = predictions_path.with_suffix("").with_suffix(".json")
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    if args.class_mapping:
        class_mapping_path = _resolve_path(repo_root, args.class_mapping)
    elif metadata.get("class_map"):
        class_mapping_path = _resolve_path(repo_root, metadata["class_map"])
    else:
        class_mapping_path = _resolve_path(repo_root, DEFAULT_CLASS_MAPPING)
    if not class_mapping_path.exists():
        print(f"ERROR: class mapping not found: {class_mapping_path}", file=sys.stderr)
        return 1

    model_name = args.model_name or str(metadata.get("model", "unknown_model")).split(".")[0]
    model_dir = taxonomy_dir / model_name
    print(f"Predictions: {predictions_path}")
    print(f"Model output dir: {model_dir}")
    print(f"Levels: {levels}")

    class_keys = _load_class_mapping(class_mapping_path)
    print(f"Classes: {len(class_keys)}")
    print("Loading taxonomy:")
    taxonomy = _load_taxonomy(class_names_path, class_keys)

    data = np.load(predictions_path)
    values, indices, labels = data["values"], data["indices"], data["labels"].astype(np.int64)
    num_classes = int(metadata.get("num_classes") or len(class_keys))
    n = len(labels)
    print(f"Images: {n}")

    # Per-species metrics serve twice: as the "species" level itself and as
    # the base of the per-group macro columns of every higher level (mean of
    # the per-species metrics over the species of each group).
    print("Computing per-species metrics...")
    species_metrics = _species_level(values, indices, labels, num_classes)

    results = []
    for level in levels:
        print(f"Computing level '{level}'...")
        if level == "species":
            result = species_metrics
        else:
            result = _grouped_level(
                level, values, indices, labels,
                taxonomy[level]["names"], taxonomy[level]["species_to_group"],
                ignore_ungrouped=args.ignore_ungrouped,
                renormalize=not args.no_renormalize,
            )
        results.append(result)
        n_species_per_group = np.bincount(taxonomy[level]["species_to_group"] if level != "species"
                                          else np.arange(num_classes, dtype=np.int64),
                                          minlength=len(result["group_names"]))
        macro = None
        if level != "species":
            macro = _species_macro_per_group(
                species_metrics, taxonomy[level]["species_to_group"],
                len(taxonomy[level]["names"]))
            # Drop trailing entries for groups removed from the result
            # (the synthetic group with --ignore-ungrouped is kept last).
            macro = {metric: arr[:len(result["group_names"])]
                     for metric, arr in macro.items()}
        _write_group_csv(model_dir / "per_group" / f"{level}.csv", result,
                         n_species_per_group, macro)
        print(f"  {level}: {len(result['group_names'])} groups, "
              f"f1_macro={result['f1'].mean():.4f}, "
              f"top1={result['top1_correct'].sum() / max(result['n_included'], 1):.4f}")

    _write_summary(model_dir / "summary.csv", results)
    _write_group_mapping(taxonomy_dir / "group_mapping.csv", class_keys, taxonomy)

    out_metadata = {
        "predictions": str(predictions_path),
        "model": metadata.get("model", "unknown_model"),
        "model_dir": model_name,
        "class_names_csv": str(class_names_path),
        "class_names_sha256": _sha256(class_names_path),
        "class_mapping": str(class_mapping_path),
        "class_mapping_sha256": _sha256(class_mapping_path),
        "levels": levels,
        "renormalized_group_scores": not args.no_renormalize,
        "ignore_ungrouped": args.ignore_ungrouped,
        "num_images": n,
        "num_classes": num_classes,
        "macro_definition": "mean over all groups at the level, zero_division=0; f1_macro_supported averages only groups with test support",
        "per_group_macro_definition": "per-group precision_macro/recall_macro/f1_macro (levels above species): mean of the per-species precision/recall/F1 over the species of the group, species with test images only (matching the training f1_macro); groups whose species all lack test images get 0",
        "created": metadata.get("created", ""),
    }
    stem = predictions_path.name
    for suffix in predictions_path.suffixes:
        stem = stem[: -len(suffix)]
    metadata_out = model_dir / f"{stem}_taxonomy_metrics.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    with metadata_out.open("w", encoding="utf-8") as handle:
        json.dump(out_metadata, handle, indent=2)

    print(f"Summary saved to {model_dir / 'summary.csv'}")
    print(f"Per-group CSVs saved to {model_dir / 'per_group'}")
    print(f"Group mapping saved to {taxonomy_dir / 'group_mapping.csv'}")
    print(f"Metadata saved to {metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
