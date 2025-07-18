#!/usr/bin/env bash
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

set -euo pipefail

echo "==> 1) Attivazione virtualenv"
source .venv/bin/activate

echo "==> 2) Caricamento moduli CUDA, compiler e cuDNN"
module unload cuda/12.1.1 || true
module load cuda/12.1.1 compiler/gcc-10.1 cudnn/v9.10.0.56

echo "==> 3) Configurazione variabili d’ambiente"
export CUDA_HOME=/storage/software/cuda/cuda-12.1.1
export CC=/storage/software/compiler/gcc-10.1.0/bin/gcc
export CXX=/storage/software/compiler/gcc-10.1.0/bin/g++
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


echo "==> 5) Build in modalità editable"
uv pip install -e .

echo "==> 6) Verifica import pointops"
echo "Testing pointops installation"
python test_pointops.py

echo "==> 7) Test import sam2._C"
python3 - << 'EOF'
import sam2
import sam2.utils
import sam2._C
print("✔ Import di sam2._C avvenuto correttamente")
EOF

echo "==> 8) Esecuzione SAM3D per generazione maschere"
uv run sam3d \
    --rgb_path data/scannetv2_images \
    --data_path preprocessed-data \
    --save_path outputs/sam2_hiera_t_iou \
    --generator_config_file configs/sam2.1/sam2_efficient_hiera_t.yaml

echo "==> Job completato con successo."