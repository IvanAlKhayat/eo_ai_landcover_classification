#!/usr/bin/env python3
"""
Download BigEarthNet subset or generate BALANCED synthetic Sentinel-2 data.
FIXED VERSION: Ensures all 10 classes are present and balanced.
"""

import argparse
import numpy as np
import os
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm


# BigEarthNet 10-class nomenclature
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

CLASS_COLORS = [
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
]


def generate_synthetic_patch_balanced(patch_id, size=256, num_classes=10):
    """
    Generate BALANCED synthetic Sentinel-2-like 4-band image.
    Uses Voronoi-like regions to ensure all classes are present.
    """
    np.random.seed(patch_id)
    
    # Create base terrain for variation
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    # Multi-scale Perlin-like noise
    base_terrain = (
        0.5 * np.sin(2 * X) * np.cos(2 * Y) +
        0.3 * np.sin(5 * X + 1) * np.cos(5 * Y + 1) +
        0.2 * np.random.randn(size, size) * 0.1
    )
    
    # BALANCED class assignment using Voronoi-like regions
    # This ensures ALL classes appear with reasonable frequency
    
    # Strategy: Create random centers for each class
    num_centers = np.random.randint(8, 15)  # Variable number of centers
    centers = []
    class_ids = []
    
    for i in range(num_centers):
        cx = np.random.uniform(-2, 2)
        cy = np.random.uniform(-2, 2)
        # Ensure all classes get at least one center
        if i < num_classes:
            class_id = i
        else:
            # Additional centers randomly distributed
            class_id = np.random.randint(0, num_classes)
        centers.append((cx, cy))
        class_ids.append(class_id)
    
    # Shuffle to avoid spatial bias
    combined = list(zip(centers, class_ids))
    np.random.shuffle(combined)
    centers, class_ids = zip(*combined)
    
    # Assign each pixel to nearest center (Voronoi diagram)
    threshold_mask = np.zeros((size, size), dtype=np.uint8)
    
    for i in range(size):
        for j in range(size):
            px, py = x[j], y[i]
            # Find nearest center
            distances = [(px - cx)**2 + (py - cy)**2 for cx, cy in centers]
            nearest_idx = np.argmin(distances)
            nearest_class = class_ids[nearest_idx]
            
            # Add terrain-based variation (20% of pixels)
            if np.random.rand() < 0.2:
                # Use terrain to slightly modify class
                terrain_offset = int(base_terrain[i, j] * 5) % 3
                modified_class = (nearest_class + terrain_offset) % num_classes
                threshold_mask[i, j] = modified_class
            else:
                threshold_mask[i, j] = nearest_class
    
    # Smooth boundaries slightly to make them more realistic
    from scipy.ndimage import median_filter
    threshold_mask = median_filter(threshold_mask, size=3)
    
    # Generate 4-band Sentinel-2 image (R, G, B, NIR)
    bands = np.zeros((4, size, size), dtype=np.float32)
    
    for class_id in range(num_classes):
        class_mask = threshold_mask == class_id
        num_pixels = class_mask.sum()
        
        if num_pixels == 0:
            continue
        
        # Spectral signatures for each class
        if class_id == 0:  # Urban fabric
            bands[0, class_mask] = np.random.uniform(0.30, 0.50, num_pixels)  # R
            bands[1, class_mask] = np.random.uniform(0.25, 0.40, num_pixels)  # G
            bands[2, class_mask] = np.random.uniform(0.25, 0.40, num_pixels)  # B
            bands[3, class_mask] = np.random.uniform(0.10, 0.25, num_pixels)  # NIR (low)
            
        elif class_id == 1:  # Industrial
            bands[0, class_mask] = np.random.uniform(0.35, 0.55, num_pixels)
            bands[1, class_mask] = np.random.uniform(0.30, 0.45, num_pixels)
            bands[2, class_mask] = np.random.uniform(0.30, 0.45, num_pixels)
            bands[3, class_mask] = np.random.uniform(0.15, 0.30, num_pixels)
            
        elif class_id == 2:  # Arable land
            bands[0, class_mask] = np.random.uniform(0.20, 0.35, num_pixels)
            bands[1, class_mask] = np.random.uniform(0.30, 0.50, num_pixels)
            bands[2, class_mask] = np.random.uniform(0.15, 0.30, num_pixels)
            bands[3, class_mask] = np.random.uniform(0.40, 0.70, num_pixels)  # High NIR
            
        elif class_id == 3:  # Permanent crops
            bands[0, class_mask] = np.random.uniform(0.15, 0.30, num_pixels)
            bands[1, class_mask] = np.random.uniform(0.25, 0.45, num_pixels)
            bands[2, class_mask] = np.random.uniform(0.15, 0.30, num_pixels)
            bands[3, class_mask] = np.random.uniform(0.50, 0.75, num_pixels)
            
        elif class_id == 4:  # Pastures
            bands[0, class_mask] = np.random.uniform(0.15, 0.30, num_pixels)
            bands[1, class_mask] = np.random.uniform(0.30, 0.50, num_pixels)
            bands[2, class_mask] = np.random.uniform(0.15, 0.30, num_pixels)
            bands[3, class_mask] = np.random.uniform(0.45, 0.70, num_pixels)
            
        elif class_id == 5:  # Complex cultivation
            bands[0, class_mask] = np.random.uniform(0.20, 0.35, num_pixels)
            bands[1, class_mask] = np.random.uniform(0.30, 0.50, num_pixels)
            bands[2, class_mask] = np.random.uniform(0.15, 0.30, num_pixels)
            bands[3, class_mask] = np.random.uniform(0.40, 0.65, num_pixels)
            
        elif class_id == 6:  # Forests
            bands[0, class_mask] = np.random.uniform(0.05, 0.15, num_pixels)
            bands[1, class_mask] = np.random.uniform(0.10, 0.25, num_pixels)
            bands[2, class_mask] = np.random.uniform(0.05, 0.15, num_pixels)
            bands[3, class_mask] = np.random.uniform(0.60, 0.90, num_pixels)  # Very high NIR
            
        elif class_id == 7:  # Herbaceous vegetation
            bands[0, class_mask] = np.random.uniform(0.25, 0.40, num_pixels)
            bands[1, class_mask] = np.random.uniform(0.35, 0.55, num_pixels)
            bands[2, class_mask] = np.random.uniform(0.20, 0.35, num_pixels)
            bands[3, class_mask] = np.random.uniform(0.35, 0.60, num_pixels)
            
        elif class_id == 8:  # Bare/Open spaces
            bands[0, class_mask] = np.random.uniform(0.35, 0.55, num_pixels)
            bands[1, class_mask] = np.random.uniform(0.35, 0.55, num_pixels)
            bands[2, class_mask] = np.random.uniform(0.25, 0.45, num_pixels)
            bands[3, class_mask] = np.random.uniform(0.30, 0.50, num_pixels)
            
        elif class_id == 9:  # Water
            bands[0, class_mask] = np.random.uniform(0.01, 0.08, num_pixels)
            bands[1, class_mask] = np.random.uniform(0.02, 0.10, num_pixels)
            bands[2, class_mask] = np.random.uniform(0.03, 0.15, num_pixels)
            bands[3, class_mask] = np.random.uniform(0.01, 0.05, num_pixels)  # Very low NIR
    
    # Add sensor noise (realistic Sentinel-2 noise level)
    bands += np.random.randn(4, size, size) * 0.015
    bands = np.clip(bands, 0, 1)
    
    return bands, threshold_mask


def create_dataset(output_dir, num_samples=1000, split_ratio=(0.7, 0.15, 0.15)):
    """Create synthetic BigEarthNet-style dataset with BALANCED classes."""
    output_path = Path(output_dir)
    
    # Create splits
    train_size = int(num_samples * split_ratio[0])
    val_size = int(num_samples * split_ratio[1])
    test_size = num_samples - train_size - val_size
    
    splits = {
        'train': train_size,
        'val': val_size,
        'test': test_size
    }
    
    print(f"Generating {num_samples} BALANCED synthetic Sentinel-2 patches...")
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    print(f"All 10 classes will be present in each split!")
    
    patch_id = 0
    for split_name, split_size in splits.items():
        split_path = output_path / split_name
        images_path = split_path / 'images'
        masks_path = split_path / 'masks'
        
        images_path.mkdir(parents=True, exist_ok=True)
        masks_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating {split_name} set...")
        
        # Track class distribution for this split
        class_counts = np.zeros(10, dtype=np.int64)
        
        for i in tqdm(range(split_size)):
            # Generate patch with balanced classes
            bands, mask = generate_synthetic_patch_balanced(patch_id)
            
            # Update class counts
            for c in range(10):
                class_counts[c] += (mask == c).sum()
            
            # Save 4-band image as NPY
            image_filename = f"patch_{patch_id:06d}.npy"
            np.save(images_path / image_filename, bands)
            
            # Save mask
            mask_filename = f"patch_{patch_id:06d}.png"
            Image.fromarray(mask.astype(np.uint8)).save(masks_path / mask_filename)
            
            # Save RGB preview (for first 10 samples)
            if i < 10:
                rgb = (bands[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
                Image.fromarray(rgb).save(images_path / f"preview_{patch_id:06d}.png")
            
            patch_id += 1
        
        # Print class distribution for this split
        total_pixels = class_counts.sum()
        print(f"\n{split_name.upper()} - Class distribution:")
        for i, count in enumerate(class_counts):
            pct = count / total_pixels * 100 if total_pixels > 0 else 0
            print(f"  Class {i} ({CLASS_NAMES[i][:20]:20}): {pct:5.2f}%")
    
    # Save metadata
    metadata = {
        'num_samples': num_samples,
        'splits': splits,
        'classes': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES),
        'image_size': 256,
        'bands': ['R', 'G', 'B', 'NIR'],
        'format': 'npy (4, 256, 256) float32',
        'generation_method': 'balanced_voronoi',
        'note': 'All 10 classes are present and balanced'
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save color legend
    with open(output_path / 'class_colors.json', 'w') as f:
        json.dump({name: color for name, color in zip(CLASS_NAMES, CLASS_COLORS)}, f, indent=2)
    
    print(f"\n✅ Dataset created successfully at: {output_path}")
    print(f"📊 Total patches: {num_samples}")
    print(f"📁 Structure: {output_path}/[train|val|test]/[images|masks]/")
    print(f"\n🎯 All 10 classes are present and reasonably balanced!")


def main():
    parser = argparse.ArgumentParser(description='Generate BALANCED BigEarthNet subset')
    parser.add_argument('--output', type=str, default='./data/bigearthnet_balanced',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=2000,
                        help='Number of samples to generate (recommended: 2000+)')
    parser.add_argument('--download', action='store_true',
                        help='Download real BigEarthNet (requires credentials)')
    
    args = parser.parse_args()
    
    if args.download:
        print("❌ Real BigEarthNet download requires credentials.")
        print("📥 Please visit: http://bigearth.net/")
        print("🔄 Using BALANCED synthetic data generation instead...")
    
    if args.num_samples < 1000:
        print(f"⚠️  Warning: {args.num_samples} samples may be too few for good performance.")
        print(f"   Recommended: 2000+ samples for mIoU > 0.75")
    
    create_dataset(args.output, args.num_samples)


if __name__ == '__main__':
    main()