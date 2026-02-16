#!/usr/bin/env python3
"""
Evaluation script: Calculate mIoU, accuracy, FPS, and generate metrics report.
"""

import argparse
import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
import json

from models.unet import get_model
from data.preprocess import get_dataloaders
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


CLASS_NAMES = [
    "Urban",
    "Industrial",
    "Arable",
    "Crops",
    "Pastures",
    "Complex",
    "Forests",
    "Herbaceous",
    "Bare",
    "Water"
]


def calculate_metrics(predictions, targets, num_classes=10):
    """Calculate comprehensive metrics"""
    
    # Flatten arrays
    preds_flat = predictions.flatten()
    targets_flat = targets.flatten()
    
    # Per-class IoU
    ious = []
    precisions = []
    recalls = []
    
    for cls in range(num_classes):
        pred_mask = preds_flat == cls
        target_mask = targets_flat == cls
        
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
        else:
            ious.append(np.nan)
        
        # Precision and recall
        if pred_mask.sum() > 0:
            precision = intersection / pred_mask.sum()
            precisions.append(precision)
        else:
            precisions.append(np.nan)
        
        if target_mask.sum() > 0:
            recall = intersection / target_mask.sum()
            recalls.append(recall)
        else:
            recalls.append(np.nan)
    
    # Mean IoU
    miou = np.nanmean(ious)
    
    # Overall accuracy
    accuracy = (preds_flat == targets_flat).sum() / len(targets_flat)
    
    # F1 scores
    f1_scores = []
    for p, r in zip(precisions, recalls):
        if not np.isnan(p) and not np.isnan(r) and (p + r) > 0:
            f1 = 2 * (p * r) / (p + r)
            f1_scores.append(f1)
        else:
            f1_scores.append(np.nan)
    
    return {
        'miou': miou,
        'accuracy': accuracy,
        'per_class_iou': ious,
        'per_class_precision': precisions,
        'per_class_recall': recalls,
        'per_class_f1': f1_scores
    }


def evaluate_model(model, dataloader, device, num_classes=10):
    """Evaluate model on entire dataset"""
    
    model.eval()
    all_preds = []
    all_targets = []
    inference_times = []
    
    print("🔍 Running evaluation...")
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = (time.time() - start_time) * 1000  # ms
            
            preds = outputs.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            inference_times.append(inference_time)
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_targets, num_classes)
    
    # Calculate FPS
    avg_inference_time = np.mean(inference_times)
    fps = 1000 / avg_inference_time
    
    metrics['avg_inference_time_ms'] = avg_inference_time
    metrics['fps'] = fps
    
    return metrics, all_preds, all_targets


def plot_confusion_matrix(preds, targets, class_names, save_path):
    """Plot confusion matrix"""
    
    # Calculate confusion matrix
    cm = confusion_matrix(
        targets.flatten(),
        preds.flatten(),
        labels=list(range(len(class_names)))
    )
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'}
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"📊 Confusion matrix saved to: {save_path}")


def plot_per_class_metrics(metrics, class_names, save_path):
    """Plot per-class IoU, precision, recall"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.arange(len(class_names))
    
    # IoU
    axes[0].bar(x, metrics['per_class_iou'])
    axes[0].set_ylabel('IoU')
    axes[0].set_title('Per-Class IoU')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].axhline(y=metrics['miou'], color='r', linestyle='--', label=f"mIoU: {metrics['miou']:.3f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Precision
    axes[1].bar(x, metrics['per_class_precision'], color='green')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Per-Class Precision')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # Recall
    axes[2].bar(x, metrics['per_class_recall'], color='orange')
    axes[2].set_ylabel('Recall')
    axes[2].set_title('Per-Class Recall')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"📊 Per-class metrics plot saved to: {save_path}")


def generate_report(metrics, class_names, output_path):
    """Generate comprehensive evaluation report"""
    
    report = []
    report.append("=" * 70)
    report.append("LAND COVER CLASSIFICATION EVALUATION REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Overall metrics
    report.append("OVERALL METRICS:")
    report.append(f"  Mean IoU (mIoU):        {metrics['miou']:.4f}")
    report.append(f"  Overall Accuracy:       {metrics['accuracy']:.4f}")
    report.append(f"  Avg Inference Time:     {metrics['avg_inference_time_ms']:.2f} ms")
    report.append(f"  FPS:                    {metrics['fps']:.2f}")
    report.append("")
    
    # Per-class metrics
    report.append("PER-CLASS METRICS:")
    report.append(f"{'Class':<20} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    report.append("-" * 70)
    
    for i, name in enumerate(class_names):
        iou = metrics['per_class_iou'][i]
        prec = metrics['per_class_precision'][i]
        rec = metrics['per_class_recall'][i]
        f1 = metrics['per_class_f1'][i]
        
        report.append(
            f"{name:<20} {iou:>8.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}"
        )
    
    report.append("=" * 70)
    
    # Print to console
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n📄 Report saved to: {output_path}")
    
    # Also save as JSON
    json_path = str(output_path).replace('.txt', '.json')

    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📄 JSON metrics saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate U-Net model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='./data/bigearthnet_subset',
                        help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device(args.device)
    model = get_model(n_channels=4, n_classes=10)
    
    #checkpoint = torch.load(args.model, map_location=device) weights_only Error!!!!
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print(f"✅ Model loaded from: {args.model}")
    
    # Load test data
    _, _, test_loader = get_dataloaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Evaluate
    metrics, preds, targets = evaluate_model(model, test_loader, device)
    
    # Generate visualizations
    plot_confusion_matrix(
        preds, targets, CLASS_NAMES,
        output_dir / 'confusion_matrix.png'
    )
    
    plot_per_class_metrics(
        metrics, CLASS_NAMES,
        output_dir / 'per_class_metrics.png'
    )
    
    # Generate report
    generate_report(
        metrics, CLASS_NAMES,
        output_dir / 'evaluation_report.txt'
    )
    
    # Check performance targets
    print("\n" + "=" * 70)
    print("PERFORMANCE TARGETS:")
    print("=" * 70)
    print(f"mIoU > 0.80:           {'✅ PASS' if metrics['miou'] > 0.80 else '❌ FAIL'} ({metrics['miou']:.4f})")
    print(f"Inference < 50ms (CPU): {'✅ PASS' if metrics['avg_inference_time_ms'] < 50 else '❌ FAIL'} ({metrics['avg_inference_time_ms']:.2f} ms)")
    print("=" * 70)


if __name__ == '__main__':
    main()
