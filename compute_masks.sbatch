#!/bin/bash
#SBATCH --job-name="sam2_masks"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=12G
#SBATCH --time=7-00:00:00 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexzirilli@gmail.com
#SBATCH --output=/storage/slurm/ziri/SegmentAnything3D/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/ziri/SegmentAnything3D/err/slurm-%j.out


source .venv/bin/activate

module load cuda/12.1.1
module load compiler/gcc-10.1

export CUDA_HOME=/storage/software/cuda/cuda-12.1.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CC=/storage/software/compiler/gcc-10.1.0/bin/gcc
export CXX=/storage/software/compiler/gcc-10.1.0/bin/g++

echo "Using CUDA at $CUDA_HOME"
echo "Using GCC at $CC"

echo "Testing pointops installation"
python test_pointops.py

echo "Running SAM3D to compute masks..."
uv run sam3d --rgb_path data/scannetv2_images --data_path preprocessed-data --save_path outputs/old_sam2 --sam_checkpoint_path checkpoints/sam2.1_hiera_tiny.pt --config_file configs/sam2.1/sam2.1_hiera_t.yaml