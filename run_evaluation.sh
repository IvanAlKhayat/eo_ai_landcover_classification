#!/bin/bash
# ============================================================================
# Simple Evaluation Script (Run on Login Node)
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "🔍 EO-AI Model Evaluation"
echo "=========================================="

# Navigate to project directory
cd ~/eo/EO-AI-Portfolio

# Activate virtual environment
source venv/bin/activate

# Check if model exists
if [ ! -f "checkpoints/best_model.pth" ]; then
    echo "❌ Model not found: checkpoints/best_model.pth"
    exit 1
fi

# Create output directories
mkdir -p evaluation predictions assets

echo -e "\n📊 Step 1: Running evaluation on test set..."
python evaluate.py \
    --model checkpoints/best_model.pth \
    --data_path ./data/bigearthnet_subset \
    --device cpu \
    --batch_size 8 \
    --output_dir ./evaluation

echo -e "\n🖼️ Step 2: Generating prediction visualizations..."
python inference.py \
    --model checkpoints/best_model.pth \
    --input_dir data/bigearthnet_subset/test/images \
    --output_dir predictions \
    --device cpu \
    --visualize

echo -e "\n📸 Step 3: Creating assets for README..."
# Copy best visualizations
cp evaluation/confusion_matrix.png assets/ 2>/dev/null || echo "⚠️  Confusion matrix not found"
cp evaluation/per_class_metrics.png assets/ 2>/dev/null || echo "⚠️  Metrics plot not found"

# Copy first 3 predictions
for i in {0..2}; do
    pred_file=$(ls predictions/*_vis.png 2>/dev/null | sed -n "$((i+1))p")
    if [ -f "$pred_file" ]; then
        cp "$pred_file" "assets/example_$((i+1)).png"
        echo "  ✅ Copied $(basename $pred_file) → assets/example_$((i+1)).png"
    fi
done

echo -e "\n=========================================="
echo "✅ Evaluation Complete!"
echo "=========================================="
echo "📁 Results:"
echo "  - Evaluation report: evaluation/evaluation_report.txt"
echo "  - Confusion matrix:  evaluation/confusion_matrix.png"
echo "  - Metrics plot:      evaluation/per_class_metrics.png"
echo "  - Predictions:       predictions/*.png"
echo "  - Assets for README: assets/*.png"
echo ""
echo "📖 Next steps:"
echo "  1. Review: cat evaluation/evaluation_report.txt"
echo "  2. Check: ls -lh assets/"
echo "  3. Upload to GitHub with these assets!"
echo "=========================================="