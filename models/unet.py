#!/usr/bin/env python3
"""
U-Net architecture for semantic segmentation of Sentinel-2 imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Use bilinear upsampling or transposed conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle input size differences (padding if needed)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 convolution for final classification"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation.
    
    Args:
        n_channels: Number of input channels (4 for Sentinel-2: R, G, B, NIR)
        n_classes: Number of output classes (10 for BigEarthNet)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
    """
    
    def __init__(self, n_channels=4, n_classes=10, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (upsampling path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final classification
        logits = self.outc(x)
        return logits
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiceLoss(nn.Module):
    """Dice loss for imbalanced segmentation tasks"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets, num_classes=10):
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
        """
        predictions = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten spatial dimensions
        predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # Calculate Dice coefficient
        intersection = (predictions * targets_one_hot).sum(dim=2)
        union = predictions.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return average Dice loss
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Combination of Cross-Entropy and Dice loss"""
    
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()
    
    def forward(self, predictions, targets):
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.ce_weight * ce + self.dice_weight * dice


def get_model(n_channels=4, n_classes=10, pretrained=False):
    """Factory function to create U-Net model"""
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=True)
    
    if pretrained:
        print("⚠️ No pretrained weights available for Sentinel-2 U-Net")
    
    num_params = model.count_parameters()
    print(f"✅ Created U-Net with {num_params:,} parameters ({num_params/1e6:.2f}M)")
    
    return model


if __name__ == '__main__':
    # Test model
    model = get_model(n_channels=4, n_classes=10)
    
    # Test forward pass
    dummy_input = torch.randn(2, 4, 256, 256)
    output = model(dummy_input)
    
    print(f"\n✅ Model test successful!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 10, 256, 256)")
    
    # Test loss
    dummy_target = torch.randint(0, 10, (2, 256, 256))
    criterion = CombinedLoss()
    loss = criterion(output, dummy_target)
    
    print(f"\nLoss test:")
    print(f"Combined loss: {loss.item():.4f}")
