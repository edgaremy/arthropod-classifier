#!/usr/bin/bash
# One-time setup of an ARM (aarch64) conda env for arthropod-classifier training.
#
# The GPU nodes (kairosgh*) are ARM (NVIDIA Grace / GH200). Your usual
# `arthropod-classifier` conda env is x86-64 and CANNOT run there.
#
# PROBLEM: ARM compute nodes have NO internet. The system conda package cache
# at /softs/arm/miniforge/26.1.0/pkgs is read-only (can't clone). And the ARM
# conda binary won't execute on the x86 login node.
#
# SOLUTION: Run this on the x86 LOGIN NODE (which has internet). We use the
# x86 conda to solve and download aarch64 packages via CONDA_SUBDIR, then
# the resulting env is ready to use on the ARM compute nodes.
#
# Run this ONCE on the LOGIN NODE:
#   cd /users/p26060/p26060rmdg/CODE/arthropod-classifier
#   bash training/setup_arm_env.sh

set -euo pipefail

# Use the x86 conda (runs on the login node; downloads aarch64 packages).
X86_CONDA="$HOME/miniconda3/bin/conda"

if [ ! -x "$X86_CONDA" ]; then
    echo "ERROR: x86 conda not found at $X86_CONDA"
    echo "This script must run on the x86 LOGIN NODE (where internet is available)."
    exit 1
fi

# Everything is created under the user's HOME (~).
ARM_ENV="$HOME/arm-envs/arthropod-classifier"
USER_PKGS="$HOME/.conda/pkgs"

mkdir -p "$USER_PKGS" "$(dirname "$ARM_ENV")"

# Idempotent: skip if the env python and torchvision already exist.
if [ -x "$ARM_ENV/bin/python" ] && [ -f "$ARM_ENV/lib/python3.14/site-packages/torchvision/__init__.py" ]; then
    echo "ARM env already set up at $ARM_ENV. Nothing to do."
    exit 0
fi

# Force conda to:
#  - solve for linux-aarch64 (the ARM platform) while running on x86
#  - fake the CUDA virtual package so the solver accepts CUDA builds
#    (the ARM nodes have CUDA 13.2; we match the system env's cuda130_generic)
#  - use ONLY the user-writable package cache (never the read-only system one)
export CONDA_SUBDIR=linux-aarch64
export CONDA_OVERRIDE_CUDA=13.2
export CONDA_PKGS_DIRS="$USER_PKGS"

echo ">> Creating ARM (aarch64) env at $ARM_ENV ..."
echo "   (downloading CUDA-enabled torch + deps from conda-forge; this takes a few minutes)"
# Pin python=3.14 and pytorch=2.10.0=*cuda* to get the cuda130_generic aarch64
# build (not the CPU-only build). torchvision matches the torch CUDA build.
"$X86_CONDA" create -y -p "$ARM_ENV" --platform linux-aarch64 -c conda-forge \
    "python=3.14" \
    "pytorch=2.10.0=*cuda*" \
    torchvision \
    scikit-learn \
    huggingface_hub \
    safetensors \
    pyyaml \
    numpy \
    pillow

echo ""
echo ">> Installed packages:"
"$X86_CONDA" list -p "$ARM_ENV" 2>/dev/null | grep -E "pytorch|torchvision|scikit-learn|huggingface|safetensors|pyyaml|numpy" || true

echo ""
echo "Done. ARM env ready at: $ARM_ENV"
echo ""
echo "NOTE: You cannot import torch here (aarch64 binaries don't run on x86)."
echo "      The env will work on the ARM GPU nodes (kairosgh*)."
echo "      Verify on an ARM node with:"
echo "        srun --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:1 --time=00:10:00 \\"
echo "          $ARM_ENV/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo ""
echo "You can now submit the training job: sbatch training/sbatch_train.sh"
