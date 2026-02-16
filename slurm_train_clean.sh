#!/bin/bash
#SBATCH --job-name=eo-train
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================================================
# NHR Alex Training Script (Clean Version)
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "🚀 EO-AI Training Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"

# Navigate to project
cd ~/eo/EO-AI-Portfolio

# Activate environment
source venv/bin/activate

# System info
echo -e "\nSystem Info:"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# Create directories
mkdir -p logs checkpoints evaluation

# Run training
echo -e "\n=========================================="
echo "🏋️ Starting Training..."
echo "=========================================="

python train.py \
    --data_path ./data/bigearthnet_subset \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --use_class_weights \
    --amp \
    --num_workers 8 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./runs

TRAIN_EXIT=$?

# Post-training tasks
if [ $TRAIN_EXIT -eq 0 ]; then
    echo -e "\n✅ Training complete!"
    
    # Evaluation
    if [ -f "checkpoints/best_model.pth" ]; then
        echo -e "\n📊 Running evaluation..."
        python evaluate.py \
            --model checkpoints/best_model.pth \
            --data_path ./data/bigearthnet_subset \
            --device cuda \
            --output_dir ./evaluation
        
        # Model compression
        echo -e "\n🗜️ Compressing model..."
        python models/quantization.py \
            --checkpoint checkpoints/best_model.pth \
            --data_path ./data/bigearthnet_subset \
            --output_dir ./checkpoints
    fi
    
    echo -e "\n=========================================="
    echo "✅ All tasks complete!"
    echo "=========================================="
else
    echo -e "\n❌ Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi