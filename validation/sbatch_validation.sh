#!/usr/bin/bash
# SBATCH script for arthropod classifier VALIDATION on cluster ARM GPU nodes
# Project: p26060rmdg
# Note: GPU nodes (kairosgh*) use ARM CPUs (NVIDIA Grace) with GH200 GPUs
#       We must use gbind for proper GPU binding on ARM
#
# Unlike training (sbatch_train.sh), timm's validate.py is single-process:
# no torch.distributed, no rendezvous, no NCCL socket setup, no LOCAL_RANK
# remapping are needed. We request one node, one task, one GPU and let gbind
# bind the process to it (remapped to cuda:0, which is the only visible GPU).

# Job configuration
#SBATCH --job-name=p26060rmdg
#SBATCH --partition=gpu
#SBATCH --nodes=1              # Single node is enough for validation
#SBATCH --ntasks=1             # 1 task (timm validate.py is single-process)
#SBATCH --ntasks-per-node=1   # 1 task per node
#SBATCH --cpus-per-task=36    # 36 CPUs for dataloader workers + main thread
#SBATCH --gres=gpu:1           # 1 GPU
#SBATCH --time=2:00:00       # Generous; ~1M test images on one GH200
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

# Create logs directory for terminal output
mkdir -p output/arthropod-classifier/logs

# Dataset on the cluster shared filesystem
DATASET_DIR=/work/datasets/arthropod
NUM_CLASSES=$(wc -l < $DATASET_DIR/class-mapping.txt | awk '{print $1}')

# Best checkpoint from training (hardcoded like sbatch_train.sh)
CHECKPOINT=output/arthropod-classifier/20260926-234704-convnextv2_base_fcmae_ft_in22k_in1k_384-384/checkpoint-98.pth.tar

# Time for log file naming
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Log file path
LOG_FILE=output/arthropod-classifier/logs/validation_${SLURM_JOB_ID}_${TIMESTAMP}.log

# Results CSV: test_<checkpoint>.csv next to the checkpoint
CKPT_NAME=$(basename "$CHECKPOINT")                     # checkpoint-98.pth.tar
RESULTS_FILE="$(dirname "$CHECKPOINT")/test_${CKPT_NAME%.pth.tar}.csv"

echo "Starting validation on ARM GPU nodes: $SLURM_JOB_NODELIST"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $DATASET_DIR (test split)"
echo "Num classes: $NUM_CLASSES"

# Sanity checks before launching
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi
if [ ! -f "$DATASET_DIR/class-mapping.txt" ]; then
    echo "ERROR: Class mapping not found at $DATASET_DIR/class-mapping.txt"
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

# HuggingFace: validation loads weights from the local checkpoint only (no
# --pretrained), so no hub access is needed. Keep the offline cache settings
# to guarantee no network attempts on compute nodes.
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1

# Verify the ARM env has the packages timm's validate.py requires
# (torch, torchvision, sklearn for f1_macro, huggingface_hub, safetensors).
$ARM_PYTHON -c "import torch, torchvision, sklearn, huggingface_hub, safetensors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: ARM env at $ARM_ENV is missing required packages."
    echo "Re-run the setup on an ARM node: bash training/setup_arm_env.sh"
    exit 1
fi

# Launch validation (single task, single GPU bound by gbind)
# --entry runs timm validate.py in-process with the arthropod f1_macro tweaks
# (f1_macro + per-class F1), mirroring what training/torchrun_validation.py
# does locally. The test split is selected with --split test.
srun --export=ALL gbind $ARM_PYTHON \
    -u validation/torchrun_validation.py \
    --entry \
    --data-dir $DATASET_DIR \
    --split test \
    --model convnextv2_base.fcmae_ft_in22k_in1k_384 \
    --num-classes $NUM_CLASSES \
    --input-size 3 384 384 \
    --class-map $DATASET_DIR/class-mapping.txt \
    --checkpoint $CHECKPOINT \
    -b 64 \
    -j 16 \
    --amp \
    --metrics-avg macro \
    --results-file $RESULTS_FILE > $LOG_FILE 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Validation FAILED (exit code $EXIT_CODE). Log: $LOG_FILE"
    exit $EXIT_CODE
fi

echo "Validation completed successfully. Log saved to $LOG_FILE"
echo "Results CSV saved to $RESULTS_FILE"
