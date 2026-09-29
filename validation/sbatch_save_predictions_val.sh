#!/usr/bin/bash
# SBATCH script for arthropod classifier PREDICTION SAVING (VAL SPLIT) on cluster ARM GPU nodes
# Project: p26060rmdg
# Note: GPU nodes (kairosgh*) use ARM CPUs (NVIDIA Grace) with GH200 GPUs
#       We must use gbind for proper GPU binding on ARM
#
# Val-split variant of sbatch_save_predictions.sh: like the test-split
# script, it stores the top-k (k=1000) class probabilities per val image
# in one sparse .npz (plus ground-truth label and image path per row), so
# later analyses never need to re-run the model.
#
# Like the metrics validation, timm's eval path is single-process: no
# torch.distributed rendezvous, no NCCL setup. One node, one task, one GPU.

# Job configuration
#SBATCH --job-name=p26060rmdg
#SBATCH --partition=gpu
#SBATCH --nodes=1              # Single node is enough for prediction saving
#SBATCH --ntasks=1             # 1 task (single-process prediction pass)
#SBATCH --ntasks-per-node=1   # 1 task per node
#SBATCH --cpus-per-task=36    # 36 CPUs for dataloader workers + main thread
#SBATCH --gres=gpu:1           # 1 GPU
#SBATCH --time=2:00:00       # Generous; ~1M val images on one GH200
#SBATCH --output=slurm-%j.out  # Output file
#SBATCH --error=slurm-%j.err   # Error file

# Set OMP threads for efficient CPU usage with PyTorch
export OMP_NUM_THREADS=8

# Load required modules for ARM GPU nodes
module purge
module load nvidia
module load nvhpc-hpcx

# Change to project directory
cd /users/p26060/p26060rmdg/CODE/arthropod-classifier

# Create logs directory for terminal output and predictions output directory
mkdir -p output/arthropod-classifier/logs
mkdir -p validation/predictions

# Dataset on the cluster shared filesystem
DATASET_DIR=/work/datasets/arthropod
NUM_CLASSES=$(wc -l < $DATASET_DIR/class-mapping.txt | awk '{print $1}')

# Best checkpoint from training (hardcoded like sbatch_train.sh)
CHECKPOINT=output/arthropod-classifier/20260926-234704-convnextv2_base_fcmae_ft_in22k_in1k_384-384/checkpoint-98.pth.tar

# Top-k probabilities to save per image
TOP_K=1000

# Time for log file naming
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Log file path
LOG_FILE=output/arthropod-classifier/logs/predictions_val_${SLURM_JOB_ID}_${TIMESTAMP}.log

# Sparse predictions .npz in validation/predictions/ (a .json metadata
# sidecar with the same stem is written by the Python script)
CKPT_NAME=$(basename "$CHECKPOINT")                     # checkpoint-98.pth.tar
PREDICTIONS_FILE="validation/predictions/val_${CKPT_NAME%.pth.tar}_top${TOP_K}_predictions.npz"

echo "Starting prediction saving on ARM GPU nodes: $SLURM_JOB_NODELIST"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $DATASET_DIR (val split)"
echo "Num classes: $NUM_CLASSES"
echo "Top-k: $TOP_K"
echo "Split: val"
echo "Predictions output: $PREDICTIONS_FILE"

# Sanity checks before launching
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi
if [ ! -f "$DATASET_DIR/class-mapping.txt" ]; then
    echo "ERROR: Class mapping not found at $DATASET_DIR/class-mapping.txt"
    exit 1
fi
if [ -e "$PREDICTIONS_FILE" ]; then
    echo "ERROR: Predictions file already exists: $PREDICTIONS_FILE"
    echo "Move or delete it first (a re-run would overwrite the artifact)."
    exit 1
fi

# Use the user-owned ARM (aarch64) conda env prepared by setup_arm_env.sh.
# It is a clone of the system pytorch-2.10.0 env (torch built for cuda130_generic
# aarch64) plus torchvision, scikit-learn, huggingface_hub, safetensors.
# NOTE: your x86 `arthropod-classifier` conda env CANNOT run on these ARM nodes.
ARM_ENV=/users/p26060/p26060rmdg/arm-envs/arthropod-classifier
ARM_PYTHON=$ARM_ENV/bin/python

# Check that the ARM env has been set up
if [ ! -x "$ARM_PYTHON" ]; then
    echo "ERROR: ARM Python not found at $ARM_PYTHON"
    echo "Run this once on the login node to create the env first:"
    echo "  bash training/setup_arm_env.sh"
    exit 1
fi

# Set environment for the ARM env
export CONDA_PREFIX=$ARM_ENV
export PYTHONPATH=/users/p26060/p26060rmdg/CODE/arthropod-classifier:$PYTHONPATH
export PATH=$ARM_ENV/bin:$PATH

# gbind configuration
export BIND_CPUNODE=0
export VERBOSE=1

# HuggingFace: predictions load weights from the local checkpoint only (no
# --pretrained), so no hub access is needed. Keep the offline cache settings
# to guarantee no network attempts on compute nodes.
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1

# Verify the ARM env has the packages the prediction saving requires
# (torch, torchvision, huggingface_hub, safetensors; sklearn is not needed
# here but is part of the env check used by the other validation scripts).
$ARM_PYTHON -c "import torch, torchvision, huggingface_hub, safetensors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: ARM env at $ARM_ENV is missing required packages."
    echo "Re-run the setup on an ARM node: bash training/setup_arm_env.sh"
    exit 1
fi

# Launch prediction saving (single task, single GPU bound by gbind)
# --entry runs the capture loop in-process, reusing timm validate.py's
# argument parser and model/dataset/loader setup so preprocessing is
# identical to the metrics validation. The val split is --split val.
srun --export=ALL gbind $ARM_PYTHON \
    -u validation/torchrun_save_predictions.py \
    --entry \
    --data-dir $DATASET_DIR \
    --split val \
    --model convnextv2_base.fcmae_ft_in22k_in1k_384 \
    --num-classes $NUM_CLASSES \
    --input-size 3 384 384 \
    --class-map $DATASET_DIR/class-mapping.txt \
    --checkpoint $CHECKPOINT \
    -b 64 \
    -j 16 \
    --amp \
    --top-k $TOP_K \
    --predictions $PREDICTIONS_FILE > $LOG_FILE 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Prediction saving FAILED (exit code $EXIT_CODE). Log: $LOG_FILE"
    exit $EXIT_CODE
fi

echo "Prediction saving completed successfully. Log saved to $LOG_FILE"
echo "Sparse predictions saved to $PREDICTIONS_FILE (+ .json metadata sidecar)"
