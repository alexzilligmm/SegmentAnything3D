#!/bin/bash
set -e

module load cuda/12.1.1
module load compiler/gcc-10.1

export CUDA_HOME=/storage/software/cuda/cuda-12.1.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CC=/storage/software/compiler/gcc-10.1.0/bin/gcc
export CXX=/storage/software/compiler/gcc-10.1.0/bin/g++

echo "Using CUDA at $CUDA_HOME"
echo "Using GCC at $CC"

export TORCH_CUDA_ARCH_LIST="8.6"

cd "$(dirname "$0")/libs/pointops"

python setup.py install