# 🚀 Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- 8GB RAM minimum
- (Optional) NVIDIA GPU with CUDA for faster training

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/EO-AI-Portfolio.git
cd EO-AI-Portfolio

# Run setup script (automated)
chmod +x setup.sh
./setup.sh

# OR manual setup:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Generate Sample Data

```bash
python data/download_bigearthnet_subset.py \
    --output ./data/bigearthnet_subset \
    --num_samples 500
```

This creates a synthetic Sentinel-2 dataset with:
- 350 training samples
- 75 validation samples
- 75 test samples

## Train Your First Model

### Single GPU (or CPU)

```bash
python train.py \
    --data_path ./data/bigearthnet_subset \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-3
```

Expected training time:
- **CPU**: ~2 hours
- **GPU (RTX 3090)**: ~15 minutes

### Multi-GPU (DDP)

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    --distributed \
    --data_path ./data/bigearthnet_subset \
    --epochs 20 \
    --batch_size 8
```

## Evaluate Performance

```bash
python evaluate.py \
    --model checkpoints/best_model.pth \
    --data_path ./data/bigearthnet_subset
```

Output:
- `evaluation/evaluation_report.txt` - Detailed metrics
- `evaluation/confusion_matrix.png` - Confusion matrix
- `evaluation/per_class_metrics.png` - Per-class performance

## Compress Model (3x Size Reduction)

```bash
python models/quantization.py \
    --checkpoint checkpoints/best_model.pth \
    --data_path ./data/bigearthnet_subset
```

Results:
- **Original**: ~31 MB
- **Quantized**: ~10 MB (3x smaller)
- **Speed**: 3.3x faster on CPU

## Run Inference

### Single Image

```bash
python inference.py \
    --model checkpoints/quantized_model.pth \
    --image data/bigearthnet_subset/test/images/patch_000001.npy \
    --output_dir predictions \
    --visualize
```

### Batch Processing

```bash
python inference.py \
    --model checkpoints/quantized_model.pth \
    --input_dir data/bigearthnet_subset/test/images \
    --output_dir predictions \
    --visualize
```

## Launch Jupyter Notebook

```bash
jupyter notebook notebooks/01_quick_demo.ipynb
```

Or open in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/EO-AI-Portfolio/blob/main/notebooks/01_quick_demo.ipynb)

## Deploy with Docker

```bash
# Build image
cd docker
docker build -t eo-inference:latest .

# Run container
docker run -d -p 8000:8000 \
    -v $(pwd)/../checkpoints:/app/checkpoints \
    --name eo-api \
    eo-inference:latest

# Test API
curl http://localhost:8000/health
```

## HPC Deployment (Slurm)

```bash
# Edit slurm_train.sh to match your cluster configuration
# Then submit job:
sbatch slurm_train.sh

# Monitor progress
tail -f logs/slurm_*.out
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
python train.py --batch_size 4  # instead of 16
```

### CUDA Not Available

Train on CPU (slower):
```bash
python train.py --device cpu
```

### Missing Dependencies

Reinstall requirements:
```bash
pip install -r requirements.txt --force-reinstall
```

## What's Next?

1. **Use Real Data**: Download BigEarthNet from http://bigearth.net/
2. **Tune Hyperparameters**: Experiment with learning rates, architectures
3. **Add Features**: Try different augmentations, loss functions
4. **Deploy**: Set up production inference pipeline
5. **Scale**: Run multi-node training on HPC clusters

## Get Help

- 📖 [Full Documentation](README.md)
- 🐛 [Report Issues](https://github.com/yourusername/EO-AI-Portfolio/issues)
- 💬 [Discussions](https://github.com/yourusername/EO-AI-Portfolio/discussions)

---

**Happy Coding! 🛰️**
