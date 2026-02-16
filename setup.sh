#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --job-name=eo_setup

# ==============================================================================
# EO-AI Portfolio Setup Script
# ==============================================================================
# This script helps you set up the environment and get started quickly.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ==============================================================================

set -e  # Exit on error

echo "=========================================="
echo "🛰️  EO-AI Portfolio Setup"
echo "=========================================="

module load python/3.12-conda
python3 --version

# Create virtual environment
echo -e "\n📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "\n⬆️  Upgrading pip..."
pip install --upgrade pip
python -m pip install --upgrade pip

# Install requirements
echo -e "\n📥 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo -e "\n📁 Creating directories..."
mkdir -p data checkpoints logs runs evaluation predictions

# Generate sample data
echo -e "\n🌍 Generating sample dataset (1000 patches)..."
python data/download_bigearthnet_subset.py --output ./data/bigearthnet_subset --num_samples 1000

# Run a quick test
echo -e "\n🧪 Running quick tests..."
python models/unet.py
echo "  ✅ Model test passed"

python data/preprocess.py
echo "  ✅ Data preprocessing test passed"

echo -e "\n=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo -e "\nNext steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Train model: python train.py --data_path ./data/bigearthnet_subset --epochs 10"
echo "  3. Run inference: python inference.py --model checkpoints/best_model.pth --image sample.npy"
echo "  4. Check the notebook: jupyter notebook notebooks/01_quick_demo.ipynb"
echo ""
echo "For more information, see README.md"
echo ""
