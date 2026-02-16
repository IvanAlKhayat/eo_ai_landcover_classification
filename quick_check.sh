#!/bin/bash
# Quick diagnostic check

echo "🔍 EO-AI Project Status Check"
echo "=========================================="

cd ~/eo/EO-AI-Portfolio

echo -e "\n📊 Dataset:"
for split in train val test; do
    count=$(ls data/bigearthnet_subset/$split/images/*.npy 2>/dev/null | wc -l)
    echo "  $split: $count samples"
done

echo -e "\n🧠 Model:"
if [ -f "checkpoints/best_model.pth" ]; then
    size=$(du -h checkpoints/best_model.pth | cut -f1)
    echo "  ✅ best_model.pth ($size)"
else
    echo "  ❌ best_model.pth not found"
fi

if [ -f "checkpoints/quantized_model.pth" ]; then
    size=$(du -h checkpoints/quantized_model.pth | cut -f1)
    echo "  ✅ quantized_model.pth ($size)"
else
    echo "  ⚠️  quantized_model.pth not found"
fi

echo -e "\n📈 Results:"
if [ -f "evaluation/evaluation_report.txt" ]; then
    miou=$(grep "mIoU" evaluation/evaluation_report.txt | head -1 | awk '{print $NF}')
    echo "  ✅ Evaluation complete (mIoU: $miou)"
else
    echo "  ⚠️  No evaluation results"
fi

echo -e "\n🖼️ Visualizations:"
pred_count=$(ls predictions/*.png 2>/dev/null | wc -l)
asset_count=$(ls assets/*.png 2>/dev/null | wc -l)
echo "  Predictions: $pred_count"
echo "  Assets: $asset_count"

echo -e "\n=========================================="