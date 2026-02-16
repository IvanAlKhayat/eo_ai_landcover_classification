#!/usr/bin/env python3
"""
Simplified training script for NHR Alex compatibility.
Supports single-GPU and multi-GPU (single node only).
No complex distributed setup required.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models.unet import get_model, CombinedLoss
from data.preprocess import get_dataloaders, calculate_class_weights


def calculate_miou(predictions, targets, num_classes=10):
    """Calculate mean Intersection over Union"""
    ious = []
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    for cls in range(num_classes):
        pred_inds = predictions == cls
        target_inds = targets == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    
    return np.nanmean(ious)


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_miou = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            miou = calculate_miou(preds, masks)
        
        total_loss += loss.item()
        total_miou += miou
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mIoU': f'{miou:.4f}'
        })
    
    avg_loss = total_loss / len(loader)
    avg_miou = total_miou / len(loader)
    
    return avg_loss, avg_miou


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_miou = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            preds = outputs.argmax(dim=1)
            miou = calculate_miou(preds, masks)
            
            total_loss += loss.item()
            total_miou += miou
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mIoU': f'{miou:.4f}'
            })
    
    avg_loss = total_loss / len(loader)
    avg_miou = total_miou / len(loader)
    
    return avg_loss, avg_miou


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for land cover classification')
    
    # Data
    parser.add_argument('--data_path', type=str, default='./data/bigearthnet_subset',
                        help='Path to dataset')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='TensorBoard log directory')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n🚀 Starting training...")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Create model
    model = get_model(n_channels=4, n_classes=10)
    model = model.to(device)
    
    # Create dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        args.data_path,
        args.batch_size,
        args.num_workers
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {args.batch_size}")
    
    # Loss and optimizer
    class_weights = None
    if args.use_class_weights:
        class_weights = calculate_class_weights(args.data_path)
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights.tolist()}")
    
    criterion = CombinedLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None
    if scaler:
        print("Using automatic mixed precision (AMP)")
    
    # Training loop
    best_miou = 0.0
    
    print(f"\n{'='*60}")
    print("Starting training loop...")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_miou = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_loss, val_miou = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('mIoU/train', train_miou, epoch)
        writer.add_scalar('mIoU/val', val_miou, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
                'val_loss': val_loss
            }
            torch.save(save_dict, checkpoint_dir / 'best_model.pth')
            print(f"  ✅ Saved best model (mIoU: {best_miou:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(save_dict, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"  💾 Saved checkpoint")
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete! Best mIoU: {best_miou:.4f}")
    print(f"{'='*60}")
    writer.close()


if __name__ == '__main__':
    main()