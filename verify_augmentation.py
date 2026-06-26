"""
verify_augmentation.py
======================
Run this BEFORE training to confirm:
  1. Indian food directory is found and classes are loaded
  2. Augmentation is visually different for minority vs majority classes
  3. Weighted sampler is working (minority classes appear more often)

Run:
    python verify_augmentation.py
"""

from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from collections import Counter

import torch
from torchvision import transforms

# ── Paste your exact paths here ──────────────────────────────
FOOD101_DIR        = r"C:\Users\Lenovo\Desktop\food project\food-101\food-101"
INDIAN_DIR         = r"C:\Users\Lenovo\Desktop\food project\indian_food"
MINORITY_THRESHOLD = 300
IMAGE_SIZE         = 300
SEED               = 42
# ─────────────────────────────────────────────────────────────

np.random.seed(SEED)
random.seed(SEED)


# ── Rebuild augmentation pipelines (copy from train script) ──

def _base_train_aug(image_size):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def _minority_train_aug(image_size):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.60, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.TrivialAugmentWide(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.12),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.20)),
    ])


def denormalize(tensor):
    """Convert normalized tensor back to viewable PIL image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype("uint8")


# ── STEP 1: Directory diagnosis ───────────────────────────────

print("\n" + "="*60)
print("  STEP 1: Directory Diagnosis")
print("="*60)

food101_path = Path(FOOD101_DIR)
indian_path  = Path(INDIAN_DIR)

print(f"\nFood-101 dir : {food101_path}")
print(f"  Exists     : {food101_path.exists()}")
if food101_path.exists():
    images_dir = food101_path / "images"
    meta_dir   = food101_path / "meta"
    print(f"  images/    : {images_dir.exists()}")
    print(f"  meta/      : {meta_dir.exists()}")
    if meta_dir.exists():
        classes_file = meta_dir / "classes.txt"
        print(f"  classes.txt: {classes_file.exists()}")
        if classes_file.exists():
            with open(classes_file) as f:
                food101_classes = [l.strip() for l in f]
            print(f"  Classes    : {len(food101_classes)}")

print(f"\nIndian food dir : {indian_path}")
print(f"  Exists        : {indian_path.exists()}")

if not indian_path.exists():
    print("\n  ⚠️  INDIAN_DIR NOT FOUND.")
    print("  Possible reasons:")
    print("  1. The folder name/path is wrong — check spelling and case")
    print("  2. The folder is somewhere else on disk")
    print("\n  Searching nearby for likely Indian food folders...")
    search_root = Path(r"C:\Users\Lenovo\Desktop\food project")
    found = []
    for p in search_root.rglob("*"):
        if p.is_dir() and any(word in p.name.lower() for word in ["indian", "food", "dish", "curry", "biryani", "dal"]):
            found.append(p)
    if found:
        print("  Candidate directories found:")
        for f in found[:10]:
            print(f"    {f}")
        print("\n  ➜ Update INDIAN_DIR in this script AND in food101_train.py to the correct path.")
    else:
        print("  No candidate directories found nearby.")
        print("  ➜ Make sure your Indian food dataset is extracted and the path is correct.")
else:
    subdirs = [d for d in indian_path.iterdir() if d.is_dir()]
    print(f"  Subdirectories (dishes): {len(subdirs)}")
    total_imgs = 0
    dish_counts = {}
    for d in subdirs:
        imgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
        dish_counts[d.name] = len(imgs)
        total_imgs += len(imgs)
    print(f"  Total images : {total_imgs}")
    print(f"\n  Per-dish image counts:")
    for dish, count in sorted(dish_counts.items(), key=lambda x: x[1]):
        flag = " ← MINORITY" if count < MINORITY_THRESHOLD else ""
        print(f"    {dish:<35} {count:>5} images{flag}")


# ── STEP 2: Show augmentation comparison ─────────────────────

print("\n" + "="*60)
print("  STEP 2: Augmentation Visual Comparison")
print("="*60)

# Find a sample image from Food-101 to demonstrate
sample_img = None
if food101_path.exists():
    for cls_dir in (food101_path / "images").iterdir():
        imgs = list(cls_dir.glob("*.jpg"))
        if imgs:
            sample_img = imgs[0]
            sample_cls = cls_dir.name
            break

if sample_img is None:
    print("No sample image found. Skipping visual check.")
else:
    print(f"\nUsing sample: {sample_img}")
    original = Image.open(sample_img).convert("RGB")

    majority_aug = _base_train_aug(IMAGE_SIZE)
    minority_aug = _minority_train_aug(IMAGE_SIZE)

    n_versions = 4
    fig, axes = plt.subplots(3, n_versions + 1, figsize=(16, 9))
    fig.suptitle("Augmentation Verification\n(Blue = Majority aug | Red = Minority aug)", fontsize=13, fontweight="bold")

    # Row 0: original
    for j in range(n_versions + 1):
        axes[0][j].imshow(original.resize((IMAGE_SIZE, IMAGE_SIZE)))
        axes[0][j].set_title("Original" if j == 0 else "")
        axes[0][j].axis("off")
    axes[0][0].set_ylabel("Original", fontsize=11, color="gray")

    # Row 1: majority augmentation
    axes[1][0].imshow(original.resize((IMAGE_SIZE, IMAGE_SIZE)))
    axes[1][0].set_ylabel("Majority\n(Food-101)", fontsize=11, color="#2563eb")
    axes[1][0].axis("off")
    for j in range(1, n_versions + 1):
        aug_img = denormalize(majority_aug(original))
        axes[1][j].imshow(aug_img)
        axes[1][j].set_title(f"Aug #{j}", color="#2563eb")
        axes[1][j].axis("off")

    # Row 2: minority augmentation
    axes[2][0].imshow(original.resize((IMAGE_SIZE, IMAGE_SIZE)))
    axes[2][0].set_ylabel("Minority\n(Indian food)", fontsize=11, color="#dc2626")
    axes[2][0].axis("off")
    for j in range(1, n_versions + 1):
        aug_img = denormalize(minority_aug(original))
        axes[2][j].imshow(aug_img)
        axes[2][j].set_title(f"Aug #{j}", color="#dc2626")
        axes[2][j].axis("off")

    plt.tight_layout()
    out = Path(r"C:\Users\Lenovo\Desktop\food project") / "augmentation_check.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n  ✅ Augmentation comparison saved → {out}")
    print("  Open that image to see minority aug is visually stronger than majority aug.")
    plt.close()


# ── STEP 3: Sampler balance check ────────────────────────────

print("\n" + "="*60)
print("  STEP 3: Sampler Balance Check (simulated)")
print("="*60)

# Simulate what the sampler does with made-up class counts
fake_counts = {0: 1000, 1: 1000, 2: 50, 3: 80, 4: 1000}  # classes 2,3 are minority
total = sum(fake_counts.values())
class_weight = {idx: total / max(count, 1) for idx, count in fake_counts.items()}
minority_sim = {2, 3}
for idx in minority_sim:
    class_weight[idx] *= 3.0

print("\n  Simulated class weights (higher = sampled more often):")
for idx, w in class_weight.items():
    tag = " ← MINORITY (3x boost)" if idx in minority_sim else ""
    print(f"    Class {idx} ({fake_counts[idx]:>5} imgs): weight = {w:.1f}{tag}")

print("\n  ➜ During training, minority classes will appear ~3x more")
print("    often per epoch relative to their raw image count.\n")


# ── STEP 4: Quick dataset load test ──────────────────────────

print("="*60)
print("  STEP 4: Quick Dataset Load Test")
print("="*60)

try:
    # Only do this if Food-101 is accessible
    if food101_path.exists():
        from torch.utils.data import DataLoader

        # Import CombinedFoodDataset from the train script
        import sys
        sys.path.insert(0, str(Path(r"C:\Users\Lenovo\Desktop\food project")))
        from food101_train import CombinedFoodDataset

        print("\nLoading train dataset (quick test)...")
        ds = CombinedFoodDataset(FOOD101_DIR, INDIAN_DIR, split="train")
        sampler = ds.make_weighted_sampler()
        loader  = DataLoader(ds, batch_size=8, sampler=sampler, num_workers=0)

        # Grab one batch
        imgs, labels = next(iter(loader))
        print(f"  ✅ Batch shape : {imgs.shape}")
        print(f"  ✅ Label range : {labels.min().item()} – {labels.max().item()}")
        print(f"  ✅ Sampler     : OK (no KeyError means the fix is working)")

        # Check minority class frequency in a few batches
        label_counts = Counter()
        for i, (_, lbls) in enumerate(loader):
            label_counts.update(lbls.tolist())
            if i >= 49:  # 50 batches = 400 samples
                break

        minority_seen    = sum(label_counts[c] for c in ds.minority_classes if c in label_counts)
        non_minority_seen = sum(v for k, v in label_counts.items() if k not in ds.minority_classes)
        total_seen = minority_seen + non_minority_seen
        print(f"\n  In 50 batches ({total_seen} samples):")
        print(f"    Minority class samples    : {minority_seen} ({100*minority_seen/total_seen:.1f}%)")
        print(f"    Non-minority class samples: {non_minority_seen} ({100*non_minority_seen/total_seen:.1f}%)")
        print(f"    (Without sampler, minority would be ~{100*sum(ds._total_counts[c] for c in ds.minority_classes)/sum(ds._total_counts.values()):.1f}%)")
        print(f"\n  ✅ If minority % above is higher than the raw %, the sampler is working.\n")

except Exception as e:
    print(f"\n  ⚠️  Dataset load test failed: {e}")
    print("  Fix the issue above first, then re-run.\n")

print("="*60)
print("  Verification complete.")
print("="*60 + "\n")