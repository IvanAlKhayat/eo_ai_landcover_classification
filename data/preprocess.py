#!/usr/bin/env python3
"""
Data preprocessing and augmentation utilities for Sentinel-2 imagery.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Sentinel2Dataset(Dataset):
    """PyTorch Dataset for Sentinel-2 land cover classification."""
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Root directory containing train/val/test splits
            split: 'train', 'val', or 'test'
            transform: Albumentations transform pipeline
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        
        # Get all image paths
        self.image_paths = sorted((self.data_dir / 'images').glob('*.npy'))
        self.mask_paths = sorted((self.data_dir / 'masks').glob('*.png'))
        
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch: {len(self.image_paths)} images, {len(self.mask_paths)} masks"
        
        print(f"📊 Loaded {len(self.image_paths)} {split} samples")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load 4-band image (R, G, B, NIR)
        image = np.load(self.image_paths[idx])  # Shape: (4, H, W)
        image = image.transpose(1, 2, 0)  # Shape: (H, W, 4)
        
        # Load mask
        mask = np.array(Image.open(self.mask_paths[idx]))  # Shape: (H, W)
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image.float(), mask.long()


def get_transforms(split='train', img_size=256):
    """Get augmentation pipeline for training/validation."""
    
    if split == 'train':
        return A.Compose([
            # 1. Geometric augmentations (safe for all channel counts)
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5,
                border_mode=0
            ),
            
            # 2. Sensor noise and atmospheric blur
            # Note: var_limit is set low assuming data is normalized (0.0-1.0)
            A.OneOf([
                A.GaussNoise(var_limit=(0.001, 0.01), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
            
            # 3. Brightness, Contrast and Gamma (Multispectral safe)
            # Replaced ColorJitter because it doesn't support 4 channels (RGB only)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=1.0
                ),
                # Gamma simulates non-linear atmospheric scattering effects
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),
            
            # 4. Final normalization and tensor conversion
            A.Normalize(
                mean=[0.0, 0.0, 0.0, 0.0],  # Applied to all 4 bands (R, G, B, NIR)
                std=[1.0, 1.0, 1.0, 1.0],
                max_pixel_value=1.0         # Set to 1.0 if .npy are already 0-1, else 10000.0
            ),
            ToTensorV2()
        ])
    else:
        # Validation/Test: only normalization and tensor conversion
        return A.Compose([
            A.Normalize(
                mean=[0.0, 0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0, 1.0],
                max_pixel_value=1.0
            ),
            ToTensorV2()
        ])


def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    """Create train/val/test dataloaders."""
    
    train_dataset = Sentinel2Dataset(
        data_dir,
        split='train',
        transform=get_transforms('train')
    )
    
    val_dataset = Sentinel2Dataset(
        data_dir,
        split='val',
        transform=get_transforms('val')
    )
    
    test_dataset = Sentinel2Dataset(
        data_dir,
        split='test',
        transform=get_transforms('test')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def calculate_class_weights(data_dir, num_classes=10):
    """Calculate inverse frequency class weights for imbalanced datasets."""
    
    train_masks_dir = Path(data_dir) / 'train' / 'masks'
    mask_paths = list(train_masks_dir.glob('*.png'))
    
    print("📊 Calculating class weights from training set...")
    class_counts = np.zeros(num_classes, dtype=np.int64)
    
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path))
        for class_id in range(num_classes):
            class_counts[class_id] += np.sum(mask == class_id)
    
    # Inverse frequency weighting
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)
    
    # Normalize
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("Class distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {count:,} pixels (weight: {class_weights[i]:.3f})")
    
    return torch.FloatTensor(class_weights)


if __name__ == '__main__':
    # Test data loading
    data_dir = './data/bigearthnet_subset'
    
    if Path(data_dir).exists():
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir,
            batch_size=4,
            num_workers=0
        )
        
        # Test batch
        images, masks = next(iter(train_loader))
        print(f"\n✅ Data loading test successful!")
        print(f"Image batch shape: {images.shape}")  # (B, 4, H, W)
        print(f"Mask batch shape: {masks.shape}")    # (B, H, W)
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Unique mask values: {torch.unique(masks).tolist()}")
        
        # Calculate class weights
        weights = calculate_class_weights(data_dir)
        print(f"\nClass weights: {weights.tolist()}")
    else:
        print(f"❌ Data directory not found: {data_dir}")
        print("Run: python data/download_bigearthnet_subset.py first")
