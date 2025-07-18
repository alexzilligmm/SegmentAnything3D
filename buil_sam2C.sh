#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# build_sam2C.sh — Compilazione e setup di SAM3D (sam2._C) con CUDA 12.1 & cuDNN
# -----------------------------------------------------------------------------

# 1) Carica i moduli CUDA, compiler e cuDNN
module unload cuda/12.1.1 || true
module load cuda/12.1.1 compiler/gcc-10.1 cudnn/v9.10.0.56

# 2) Configura variabili d’ambiente
export CUDA_HOME=/storage/software/cuda/cuda-12.1.1
export CC=/storage/software/compiler/gcc-10.1.0/bin/gcc
export CXX=/storage/software/compiler/gcc-10.1.0/bin/g++
export PATH=$CUDA_HOME/bin:$PATH

# 3) Forza ABI legacy (coerente con PyTorch build)
export TORCH_CXX11_ABI=0

# 4) Installa PyTorch/CUDA-12.1 nel virtualenv
#    (necessario affinché il build backend trovi il modulo torch)
pip install --upgrade pip
pip uninstall -y torch torchvision torchaudio || true
pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio \
  -f https://download.pytorch.org/whl/cu121/torch_stable.html

# 5) Compila e installa l’estensione in modalità editable senza isolamento
uv pip install -e . --no-build-isolation

# 6) Verifica import di sam2._C
python3 - << 'EOF'
import torch
print(f"Torch CUDA version: {torch.version.cuda}")
import sam2._C
print("✔ Import di sam2._C avvenuto correttamente")
EOF

echo "==> Build e verifica completate con successo."