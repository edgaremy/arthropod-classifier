#!/usr/bin/bash
# SBATCH script for F1-macro threshold analysis on cluster CPU nodes
# Project: p26060rmdg
#
# Post-processing of the saved sparse predictions: sweeps a minimum
# confidence threshold (0 to 1) and plots F1-macro vs threshold.
#
# This is pure CPU/numpy/matplotlib work on the x86 CPU partition: no GPU,
# no ARM conda env needed - unlike the GPU validation scripts, we use the
# regular x86 conda env. The repo (and the predictions file) already live
# on GPFS ($WORK), as recommended for I/O-heavy calculations.

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
mkdir -p validation/metrics/F1-macro/plots

# Sparse predictions file produced by sbatch_save_predictions.sh
PREDICTIONS_FILE="validation/predictions/convnextv2_base_top1000_predictions.npz"

# Time for log file naming
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE=output/arthropod-classifier/logs/f1macro_threshold_${SLURM_JOB_ID}_${TIMESTAMP}.log

echo "Starting F1-macro threshold analysis on CPU nodes: $SLURM_JOB_NODELIST"
echo "Predictions: $PREDICTIONS_FILE"

# Sanity checks before launching
if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "ERROR: Predictions file not found: $PREDICTIONS_FILE"
    echo "Run sbatch_save_predictions.sh first."
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
$X86_PYTHON -c "import numpy, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: x86 env at $X86_ENV is missing numpy/matplotlib."
    echo "Install them with: $X86_ENV/bin/pip install numpy matplotlib"
    exit 1
fi

# Launch the analysis (single process; the 16 allocated task slots are more
# than enough, the sweep is a few minutes of numpy)
srun -n 1 $X86_PYTHON -u validation/metrics/F1-macro_threshold.py \
    --predictions $PREDICTIONS_FILE \
    > $LOG_FILE 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "F1-macro threshold analysis FAILED (exit code $EXIT_CODE). Log: $LOG_FILE"
    exit $EXIT_CODE
fi

echo "F1-macro threshold analysis completed. Log saved to $LOG_FILE"
echo "Plot saved to validation/metrics/F1-macro/plots/"
