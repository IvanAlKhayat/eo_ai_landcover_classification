#!/bin/bash
#SBATCH --job-name=eo-ai-train
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1              # 1 GPU (change to :2 or :4 for multi-GPU)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Increase for data loading
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ============================================================================
# NHR Alex Optimized Training Script
# ============================================================================
# Single-node training (multi-GPU if available)
# No module system required
# ============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Create log directory
mkdir -p logs checkpoints
module load python/3.12-conda

# Activate virtual environment (adjust path if needed)
source venv/bin/activate

# Check Python and PyTorch
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set environment for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Detect number of GPUs
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "Detected $NUM_GPUS GPU(s)"

#python3 diagnose.py

# Vai nella directory del progetto
cd ~/eo/EO-AI-Portfolio

# Attiva venv
source venv/bin/activate

# Verifica che il modello esista
ls -lh checkpoints/best_model.pth

# Esegui evaluation
python evaluate.py \
    --model checkpoints/best_model.pth \
    --data_path ./data/bigearthnet_subset \
    --device cuda \
    --output_dir ./evaluation
## Run training
#if [ "$NUM_GPUS" -gt 1 ]; then
#    echo "Running multi-GPU training with $NUM_GPUS GPUs..."
#    
#    # Multi-GPU on single node (no need for distributed setup)
#    python train.py \
#        --data_path ./data/bigearthnet_subset \
#        --epochs 50 \
#        --batch_size 16 \
#        --lr 1e-3 \
#        --weight_decay 1e-4 \
#        --use_class_weights \
#        --amp \
#        --num_workers 4 \
#        --checkpoint_dir ./checkpoints \
#        --log_dir ./runs
#else
#    echo "Running single-GPU training..."
#    
#    # Single GPU
#    python train.py \
#        --data_path ./data/bigearthnet_subset \
#        --epochs 800 \
#        --batch_size 32 \
#        --lr 1e-3 \
#        --weight_decay 1e-4 \
#        --use_class_weights \
#        --amp \
#        --num_workers 4 \
#        --checkpoint_dir ./checkpoints \
#        --log_dir ./runs
#fi
#
#TRAINING_EXIT_CODE=$?
#
#echo "=========================================="
#if [ $TRAINING_EXIT_CODE -eq 0 ]; then
#    echo "✅ Training complete!"
#    
#    # Run evaluation if model exists
#    if [ -f "checkpoints/best_model.pth" ]; then
#        echo "Running evaluation..."
#        python evaluate.py \
#            --model checkpoints/best_model.pth \
#            --data_path ./data/bigearthnet_subset \
#            --device cuda
#        
#        # Run quantization
#        echo "Compressing model..."
#        python models/quantization.py \
#            --checkpoint checkpoints/best_model.pth \
#            --data_path ./data/bigearthnet_subset \
#            --output_dir ./checkpoints
#    fi
#    
#    echo "All tasks complete!"
#else
#    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE"
#fi
#echo "=========================================="
#
## Optional: Run evaluation on best model
#if [ -f "checkpoints/best_model.pth" ]; then
#    echo "Running evaluation on best model..."
#    python evaluate.py \
#        --model checkpoints/best_model.pth \
#        --data_path ./data/bigearthnet_subset
#fi
#
## Optional: Compress model
#if [ -f "checkpoints/best_model.pth" ]; then
#    echo "Compressing model..."
#    python models/quantization.py \
#        --checkpoint checkpoints/best_model.pth \
#        --data_path ./data/bigearthnet_subset \
#        --output_dir ./checkpoints
#fi
#
#echo "All tasks complete! Check logs/ and checkpoints/ directories."
