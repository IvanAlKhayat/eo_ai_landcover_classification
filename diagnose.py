import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

data_dir = Path('data/bigearthnet_subset')  # Modifica il path se diverso

print("="*60)
print("🔍 DIAGNOSTIC REPORT")
print("="*60)

# 1. Dataset size
train_imgs = list((data_dir / 'train' / 'images').glob('*.npy'))
val_imgs = list((data_dir / 'val' / 'images').glob('*.npy'))
test_imgs = list((data_dir / 'test' / 'images').glob('*.npy'))

print(f"\n📊 Dataset Sizes:")
print(f"  Train: {len(train_imgs)}")
print(f"  Val: {len(val_imgs)}")
print(f"  Test: {len(test_imgs)}")
print(f"  TOTAL: {len(train_imgs) + len(val_imgs) + len(test_imgs)}")

# 2. Class distribution (IMPORTANTE!)
print(f"\n🎯 Class Distribution (train set):")
class_counts = np.zeros(10, dtype=np.int64)

for mask_path in tqdm(list((data_dir / 'train' / 'masks').glob('*.png')), desc="Analyzing masks"):
    mask = np.array(Image.open(mask_path))
    for i in range(10):
        class_counts[i] += (mask == i).sum()

total = class_counts.sum()
print(f"\n{'Class':<15} {'Pixels':>15} {'Percentage':>12}")
print("-"*45)
for i, count in enumerate(class_counts):
    pct = count / total * 100
    bar = "█" * int(pct / 2)  # Visual bar
    print(f"Class {i:<8} {count:>15,} {pct:>11.2f}% {bar}")

# 3. Imbalance check
max_class = class_counts.max()
min_class = class_counts[class_counts > 0].min()
max_class_id = class_counts.argmax()
min_class_id = class_counts[class_counts > 0].argmin()
ratio = max_class / min_class

print(f"\n⚖️  Imbalance Analysis:")
print(f"  Most common class: {max_class_id} ({max_class / total * 100:.1f}%)")
print(f"  Least common class: {min_class_id} ({min_class / total * 100:.1f}%)")
print(f"  Imbalance ratio: {ratio:.1f}:1")

if ratio > 100:
    print(f"  🔴 CRITICAL: Severe imbalance! Model will just predict class {max_class_id}")
elif ratio > 50:
    print(f"  🟠 WARNING: High imbalance, use class weights")
elif ratio > 10:
    print(f"  🟡 MODERATE: Imbalance present, class weights recommended")
else:
    print(f"  ✅ GOOD: Relatively balanced")

# 4. Sample diversity
print(f"\n🔬 Checking sample diversity...")
sample_masks = []
for mask_path in list((data_dir / 'train' / 'masks').glob('*.png'))[:100]:
    mask = np.array(Image.open(mask_path))
    sample_masks.append(mask)

# Check if samples are too similar
unique_class_combinations = set()
for mask in sample_masks:
    classes_present = tuple(sorted(np.unique(mask)))
    unique_class_combinations.add(classes_present)

print(f"  Unique class combinations in first 100 samples: {len(unique_class_combinations)}")
print(f"  Examples: {list(unique_class_combinations)[:5]}")

if len(unique_class_combinations) < 20:
    print(f"  ⚠️  LOW DIVERSITY: Samples are too similar")
else:
    print(f"  ✅ GOOD: Samples have variety")

# 5. Image statistics per channel
print(f"\n📸 Image Statistics (per channel):")
sample_imgs = [np.load(p) for p in train_imgs[:100]]
all_imgs = np.stack(sample_imgs)  # (100, 4, 256, 256)

for i in range(4):
    channel = all_imgs[:, i, :, :]
    print(f"  Channel {i} (R/G/B/NIR[{i}]): mean={channel.mean():.3f}, std={channel.std():.3f}, min={channel.min():.3f}, max={channel.max():.3f}")

print("\n" + "="*60)
print("💡 RECOMMENDATIONS:")
print("="*60)

# Recommendations based on findings
if len(train_imgs) < 500:
    print("⚠️  Dataset too small (<500 samples)")
    print("   → Generate more data: --num_samples 2000")
    
if ratio > 50:
    print("⚠️  Severe class imbalance detected")
    print("   → Use --use_class_weights flag in training")
    print("   → OR regenerate data with balanced classes")

if len(unique_class_combinations) < 20:
    print("⚠️  Low sample diversity")
    print("   → Dataset is too repetitive")
    print("   → Consider using real BigEarthNet data")

print("="*60)