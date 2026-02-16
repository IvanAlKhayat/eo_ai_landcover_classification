#!/bin/bash
#SBATCH --job-name=eo-eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ============================================================================
# SLURM Evaluation Script (Optional - GPU Acceleration)
# ============================================================================

echo "=========================================="
echo "🔍 EO-AI Model Evaluation (GPU)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"

# Navigate to project
cd ~/eo/EO-AI-Portfolio

# Activate environment
source venv/bin/activate

# Check GPU
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Create directories
mkdir -p evaluation predictions

# Run evaluation
echo -e "\n📊 Running evaluation..."
python evaluate.py \
    --model checkpoints/best_model.pth \
    --data_path ./data/bigearthnet_subset \
    --device cuda \
    --batch_size 16 \
    --output_dir ./evaluation

# Generate predictions
echo -e "\n🖼️ Generating predictions..."
python inference.py \
    --model checkpoints/best_model.pth \
    --input_dir data/bigearthnet_subset/test/images \
    --output_dir predictions \
    --device cuda \
    --visualize

echo -e "\n✅ Evaluation complete!"
ls -lh evaluation/