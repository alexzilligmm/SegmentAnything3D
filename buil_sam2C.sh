#!/usr/bin/env bash

set -euo pipefail

module unload cuda/12.1.1 || true
module load cuda/12.1.1 compiler/gcc-10.1 cudnn/v9.10.0.56

export CUDA_HOME=/storage/software/cuda/cuda-12.1.1
export CC=/storage/software/compiler/gcc-10.1.0/bin/gcc
export CXX=/storage/software/compiler/gcc-10.1.0/bin/g++
export PATH=$CUDA_HOME/bin:$PATH

export TORCH_LIB_DIR=/storage/slurm/ziri/SegmentAnything3D/.venv/lib/python3.10/site-packages/torch/lib

export LD_LIBRARY_PATH=$TORCH_LIB_DIR:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

uv pip install -e .

python3 - << 'EOF'
import sam2
import sam2.utils
import sam2._C
print("✔ Import di sam2._C avvenuto correttamente")
EOF
