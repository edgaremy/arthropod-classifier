#!/usr/bin/bash
# SBATCH script for taxonomy-level metrics on cluster CPU nodes
# Project: p26060rmdg
#
# Post-processing of the saved sparse predictions: computes top-1/top-5
# accuracy and per-group precision/recall/F1 at five taxonomic levels
# (species, genus, family, order, class), by summing the stored species
# probabilities per group (class-names.csv provides the taxonomy).
# Per-group CSVs at genus/family/order/class levels additionally report
# precision_macro/recall_macro/f1_macro: the mean of the per-species
# precision/recall/F1 over the species belonging to each group (e.g. a
# family row averages the per-species metrics of its species), averaging
# only species with test images (training f1_macro convention).
# Outputs go to validation/metrics/taxonomy/<model>/.
#
# This is pure CPU/numpy work on the x86 CPU partition: no GPU, no ARM
# conda env - we use the regular x86 conda env. The repo and the
# predictions file already live on GPFS ($WORK), as recommended for
# I/O-heavy calculations.

# Job configuration - MicroShared CPU template (<= 16 cpus, non-exclusive)
#SBATCH --job-name=p26060rmdg    # Nom du job        (-J JobName)
#SBATCH --nodes=1                # Nombre de noeuds  (-N 1)
#SBATCH --ntasks=16              # Nombre de taches  (-n 16)
#SBATCH --mem=64000M             # regle de 4 G par core
#SBATCH --ntasks-per-node=16     # Taches par noeud
#SBATCH --cpus-per-task=1        # Nombre de coeurs CPU par tache
#SBATCH --time=01:00:00          # Limite de temps  (-t 01:00:00)
#SBATCH --partition=micro-cpu     # Partition CPU
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=96
#SBATCH --output=slurm-%j.out    # Output file
#SBATCH --error=slurm-%j.err     # Error file

module purge

# Change to project directory (on GPFS)
cd /users/p26060/p26060rmdg/CODE/arthropod-classifier

# Create output directories
mkdir -p output/arthropod-classifier/logs
mkdir -p validation/metrics/taxonomy

# Dataset taxonomy files and sparse predictions file
DATASET_DIR=/work/datasets/arthropod
CLASS_NAMES=$DATASET_DIR/class-names.csv
CLASS_MAPPING=$DATASET_DIR/class-mapping.txt
PREDICTIONS_FILE="validation/predictions/convnextv2_base_top1000_predictions.npz"

# Taxonomy options (see validation/metrics/taxonomy_metrics.py --help):
# - Group scores are renormalized per row by default (the stored top-1000
#   species probabilities are a truncated softmax; see the script).
# - Species without a taxon (e.g. empty `order`) form a synthetic
#   "(no <level>)" group by default. Add --ignore-ungrouped to exclude
#   images whose true species has no group at a level instead (the number
#   excluded is reported per level in summary.csv).
TAXONOMY_ARGS=""

# Time for log file naming
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE=output/arthropod-classifier/logs/taxonomy_metrics_${SLURM_JOB_ID}_${TIMESTAMP}.log

echo "Starting taxonomy metrics on CPU nodes: $SLURM_JOB_NODELIST"
echo "Predictions: $PREDICTIONS_FILE"
echo "Class names: $CLASS_NAMES"

# Sanity checks before launching
if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "ERROR: Predictions file not found: $PREDICTIONS_FILE"
    echo "Run sbatch_save_predictions.sh first."
    exit 1
fi
if [ ! -f "$CLASS_NAMES" ]; then
    echo "ERROR: class-names.csv not found: $CLASS_NAMES"
    exit 1
fi

# Regular x86 conda env (the ARM env is NOT needed for CPU post-processing)
X86_ENV=/users/p26060/p26060rmdg/miniconda3/envs/arthropod-classifier
X86_PYTHON=$X86_ENV/bin/python

if [ ! -x "$X86_PYTHON" ]; then
    echo "ERROR: x86 Python not found at $X86_PYTHON"
    exit 1
fi

# Verify the env has the packages the analysis requires
$X86_PYTHON -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: x86 env at $X86_ENV is missing numpy."
    exit 1
fi

# Launch the analysis (single process; the sweep over all five levels takes
# roughly 15-25 minutes for ~1M images)
srun -n 1 $X86_PYTHON -u validation/metrics/taxonomy_metrics.py \
    --predictions $PREDICTIONS_FILE \
    --class-names $CLASS_NAMES \
    --class-mapping $CLASS_MAPPING \
    --taxonomy-dir validation/metrics/taxonomy \
    $TAXONOMY_ARGS > $LOG_FILE 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Taxonomy metrics FAILED (exit code $EXIT_CODE). Log: $LOG_FILE"
    exit $EXIT_CODE
fi

echo "Taxonomy metrics completed. Log saved to $LOG_FILE"
echo "Outputs saved under validation/metrics/taxonomy/<model>/"
