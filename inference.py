#!/usr/bin/env python3
"""
Inference script for land cover classification using quantized U-Net.
Supports batch processing and visualization.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import time

from models.unet import get_model
from data.preprocess import get_transforms


# Class names and colors from BigEarthNet
CLASS_NAMES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Forests",
    "Herbaceous vegetation",
    "Open spaces with little or no vegetation",
    "Wetlands and water bodies"
]

CLASS_COLORS = np.array([
    [230, 0, 0],      # Urban - red
    [180, 0, 0],      # Industrial - dark red
    [255, 255, 0],    # Arable - yellow
    [240, 150, 0],    # Permanent crops - orange
    [150, 255, 0],    # Pastures - light green
    [200, 200, 0],    # Complex cultivation - olive
    [0, 150, 0],      # Forests - dark green
    [150, 200, 150],  # Herbaceous - light green
    [200, 200, 200],  # Bare - gray
    [0, 100, 200]     # Water - blue
], dtype=np.uint8)


def load_model(checkpoint_path, device='cpu'):
    """Load trained or quantized model"""
    model = get_model(n_channels=4, n_classes=10)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    return model


def load_sentinel2_image(image_path):
    """Load Sentinel-2 4-band image (R, G, B, NIR)"""
    if image_path.suffix == '.npy':
        # Load from NPY
        image = np.load(image_path)  # Shape: (4, H, W)
        image = image.transpose(1, 2, 0)  # Shape: (H, W, 4)
    else:
        raise ValueError(f"Unsupported file format: {image_path.suffix}")
    
    return image


def predict_single_image(model, image, device='cpu'):
    """Run inference on single image"""
    # Prepare image
    transform = get_transforms('test')
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)  # (1, 4, H, W)
    
    # Run inference
    with torch.no_grad():
        start_time = time.time()
        output = model(image_tensor)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        prediction = output.argmax(dim=1).squeeze(0).cpu().numpy()
    
    return prediction, inference_time


def colorize_mask(mask, class_colors=CLASS_COLORS):
    """Convert class indices to RGB visualization"""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(class_colors)):
        colored[mask == class_id] = class_colors[class_id]
    
    return colored


def visualize_prediction(image, prediction, save_path=None):
    """Visualize RGB image and prediction side-by-side"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # RGB composite (first 3 bands)
    rgb = image[:, :, :3]
    rgb_vis = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # Normalize to [0, 1]
    axes[0].imshow(rgb_vis)
    axes[0].set_title('Sentinel-2 RGB Composite')
    axes[0].axis('off')
    
    # Prediction
    colored_pred = colorize_mask(prediction)
    axes[1].imshow(colored_pred)
    axes[1].set_title('Land Cover Prediction')
    axes[1].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CLASS_COLORS[i]/255, label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ]
    axes[1].legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=8
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_prediction(prediction, output_path):
    """Save prediction as PNG"""
    colored = colorize_mask(prediction)
    Image.fromarray(colored).save(output_path)
    print(f"💾 Prediction saved to: {output_path}")


def batch_inference(model, input_dir, output_dir, device='cpu', visualize=True):
    """Run inference on all images in directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(input_path.glob('*.npy'))
    
    if len(image_files) == 0:
        print(f"❌ No .npy files found in {input_dir}")
        return
    
    print(f"\n🚀 Processing {len(image_files)} images...")
    
    total_time = 0
    
    for img_path in image_files:
        # Load image
        image = load_sentinel2_image(img_path)
        
        # Predict
        prediction, inference_time = predict_single_image(model, image, device)
        total_time += inference_time
        
        # Save prediction
        pred_path = output_path / f"{img_path.stem}_pred.png"
        save_prediction(prediction, pred_path)
        
        # Visualize
        if visualize:
            vis_path = output_path / f"{img_path.stem}_vis.png"
            visualize_prediction(image, prediction, vis_path)
        
        print(f"  ✓ {img_path.name}: {inference_time:.2f} ms")
    
    avg_time = total_time / len(image_files)
    fps = 1000 / avg_time
    
    print(f"\n📊 Batch inference complete!")
    print(f"  Average inference time: {avg_time:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on Sentinel-2 imagery')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str,
                        help='Path to single image (.npy)')
    parser.add_argument('--input_dir', type=str,
                        help='Directory containing images for batch processing')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Output directory for predictions')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run inference on')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device)
    model = load_model(args.model, device)
    
    if args.image:
        # Single image inference
        print(f"\n🖼️ Processing single image: {args.image}")
        image = load_sentinel2_image(Path(args.image))
        prediction, inference_time = predict_single_image(model, image, device)
        
        # Save outputs
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pred_path = output_path / 'prediction.png'
        save_prediction(prediction, pred_path)
        
        if args.visualize:
            vis_path = output_path / 'visualization.png'
            visualize_prediction(image, prediction, vis_path)
        
        print(f"\n⏱️ Inference time: {inference_time:.2f} ms")
        print(f"🚀 FPS: {1000/inference_time:.2f}")
        
    elif args.input_dir:
        # Batch inference
        batch_inference(
            model,
            args.input_dir,
            args.output_dir,
            device,
            args.visualize
        )
    
    else:
        print("❌ Please provide either --image or --input_dir")


if __name__ == '__main__':
    main()
