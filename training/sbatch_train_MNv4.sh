#!/usr/bin/bash
# SBATCH script for arthropod classifier training on cluster ARM GPU nodes
# Project: p26060rmdg
# Note: GPU nodes (kairosgh*) use ARM CPUs (NVIDIA Grace) with GH200 GPUs
#       We must use gbind for proper GPU binding on ARM

# Job configuration
#SBATCH --job-name=p26060rmdg
#SBATCH --partition=gpu
#SBATCH --nodes=2              # Use 2 nodes (gpu partition)
#SBATCH --ntasks=8             # 8 tasks total (4 GPUs per node x 2 nodes)
#SBATCH --ntasks-per-node=4   # 4 tasks per node (one per GPU)
#SBATCH --cpus-per-task=36    # 36 CPUs per task
#SBATCH --gres=gpu:4           # Request 4 GPUs PER NODE (total 8 GPUs for 2 nodes)
#SBATCH --time=36:00:00       # Maximum time for gpu partition
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

# Get the first node IP address for master address
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_IP=$(getent ahostsv4 $MASTER_NODE | awk '{print $1}' | head -n 1)

# Fallback: try different methods to get IP
if [ -z "$MASTER_IP" ]; then
    MASTER_IP=$(getent hosts $MASTER_NODE | awk '{print $1}')
fi
if [ -z "$MASTER_IP" ]; then
    echo "ERROR: Could not resolve $MASTER_NODE to IP"
    exit 1
fi

# Generate a random port for distributed communication
MASTER_PORT=$((25000 + RANDOM % 5000))

# Set the number of classes
NUM_CLASSES=$(wc -l < /work/datasets/arthropod/class-mapping.txt | awk '{print $1}')

# Time for log file naming
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Log file path
LOG_FILE=output/arthropod-classifier/logs/training_${SLURM_JOB_ID}_${TIMESTAMP}.log

echo "Starting distributed training on ARM GPU nodes: $SLURM_JOB_NODELIST"
echo "Master IP: $MASTER_IP"
echo "Master port: $MASTER_PORT"

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

# Network configuration for ARM GPU nodes.
# mlx5_* are RDMA device names, NOT socket interfaces - NCCL's TCP bootstrap
# needs a real IP interface. Auto-detect the interface that routes to the
# master node's IP, excluding loopback (lo).
MASTER_IFACE=$(ip -o -4 route get $MASTER_IP 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -1)
if [ "$MASTER_IFACE" = "lo" ] || [ -z "$MASTER_IFACE" ]; then
    echo "WARNING: Auto-detected '$MASTER_IFACE' for $MASTER_IP, falling back to ^lo (any non-loopback)"
    MASTER_IFACE="^lo"
fi
echo "Using network interface: $MASTER_IFACE (routes to $MASTER_IP)"
export NCCL_SOCKET_IFNAME=$MASTER_IFACE
export GLOO_SOCKET_IFNAME=$MASTER_IFACE

# Set distributed environment variables for PyTorch.
# This timm version uses env:// init_method (no --dist-url flag), so it reads
# MASTER_ADDR and MASTER_PORT from the environment.
export WORLD_SIZE=8
export MASTER_ADDR=$MASTER_IP
export MASTER_PORT=$MASTER_PORT

# gbind binds each task to one physical GPU and remaps it to cuda:0.
# timm reads LOCAL_RANK (before SLURM_LOCALID) to pick the device index, so
# if we let it use SLURM_LOCALID (0-3), it tries cuda:1/2/3 which don't exist
# (only cuda:0 is visible per task). Setting LOCAL_RANK=0 makes timm use cuda:0,
# which is the gbind-bound GPU. Global rank still comes from SLURM_PROCID.
export LOCAL_RANK=0

# HuggingFace pretrained weights: pre-downloaded on the login node (no internet
# on compute nodes). HF_HOME points to the shared NFS cache; HF_HUB_OFFLINE=1
# prevents any network attempts.
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1

# Verify the ARM env has the packages timm's train.py requires.
# (No on-the-fly install here: that belongs in setup_arm_env.sh, run once.)
$ARM_PYTHON -c "import torch, torchvision, sklearn, huggingface_hub, safetensors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: ARM env at $ARM_ENV is missing required packages."
    echo "Re-run the setup on an ARM node: bash training/setup_arm_env.sh"
    exit 1
fi

# Launch distributed training
# Use srun to launch 8 processes (one per GPU)
# Each process is wrapped with gbind for GPU binding
# We use --export=ALL to pass all environment variables to srun
srun --export=ALL gbind $ARM_PYTHON \
    -u src/train_arthropod.py \
    --data-dir /work/datasets/arthropod \
    --model timm/mobilenetv4_hybrid_medium.ix_e550_r384_in1k \
    --pretrained \
    --input-size 3 384 384 \
    --class-map /work/datasets/arthropod/class-mapping.txt \
    --num-classes $NUM_CLASSES \
    --epochs 100 \
    -b 32 \
    -vb 64 \
    -j 16 \
    --log-interval 200 \
    --opt lamb \
    --lr 3e-4 \
    --sched cosine \
    --weight-decay 0.01 \
    --warmup-epochs 5 \
    --smoothing 0.1 \
    --drop-path 0.05 \
    --mixup 0.2 \
    --cutmix 1.0 \
    --hflip 0.5 \
    --aa rand-m7-mstd0.5 \
    --bce-loss \
    --amp \
    --eval-metric f1_macro \
    --output output/arthropod-classifier > $LOG_FILE 2>&1 \
    --checkpoint-hist 5 \
    --resume output/arthropod-classifier/20260923-100748-mobilenetv4_hybrid_medium_ix_e550_r384_in1k-384/checkpoint-16.pth.tar \
    --start-epoch 17 > $LOG_FILE 2>&1

echo "Training completed successfully. Log saved to $LOG_FILE"
