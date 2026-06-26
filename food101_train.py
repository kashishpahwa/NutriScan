"""
Combined Food-101 + Indian Food Classification Training Pipeline
================================================================
Your Indian food dishes are already merged into the Food-101 images/ folder
(confirmed: 182 classes, includes biryani, chicken_curry, etc.)

This version additionally writes `augmentation_config.json` next to your
checkpoint, snapshotting exactly which classes were treated as minority and
which augmentation pipeline was used — a permanent, inspectable record of
what augmentation ran to produce this trained model.

Note: augmentation itself is intentionally applied LIVE every epoch (a
different random transform per image, per epoch). That is what makes
augmentation effective — baking it into static saved files would mean the
model just memorizes a fixed set of copies instead of learning invariances.
"""

import os, time, json, copy, random, shutil
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

# ─────────────────────────────────────────────
#  ✏️  EDIT THIS SECTION TO CONFIGURE TRAINING
# ─────────────────────────────────────────────
MODEL       = "efficientnet_b3"
FOOD101_DIR = r"C:\Users\Lenovo\Desktop\food project\food-101\food-101"
INDIAN_DIR  = ""   # already merged into Food-101 images folder — leave blank
EPOCHS      = 30
BATCH_SIZE  = 32
LR          = 3e-5
SAVE_DIR    = r"C:\Users\Lenovo\Desktop\food project\food-101\result"
UTILS_DIR   = r"C:\Users\Lenovo\Desktop\food project\utils"
NUM_WORKERS = 0
TRAIN_RATIO = 0.90
MINORITY_THRESHOLD = 150   # classes below this image count get heavy augmentation + boosted sampling
MINORITY_BOOST      = 3.0  # sampler weight multiplier for minority classes
# ─────────────────────────────────────────────

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else
                           "mps"  if torch.backends.mps.is_available() else "cpu")
IMAGE_SIZE = 300 if MODEL == "efficientnet_b3" else 224
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ─────────────────────────────────────────────
#  AUGMENTATION PIPELINES
# ─────────────────────────────────────────────

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
    """Heavier augmentation for under-represented (Indian food) classes."""
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


def _val_transform(image_size):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────

class CombinedFoodDataset(Dataset):
    def __init__(self, food101_dir, indian_dir, split="train",
                 train_ratio=TRAIN_RATIO, minority_threshold=MINORITY_THRESHOLD, seed=SEED):
        self.split = split

        all_samples_by_class = {}

        food101_dir = Path(food101_dir)
        with open(food101_dir / "meta" / "classes.txt") as f:
            food101_classes = [l.strip() for l in f]

        self.classes = list(food101_classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in food101_classes:
            img_dir = food101_dir / "images" / cls
            paths   = [str(p) for p in img_dir.glob("*.jpg") if p.exists()]
            paths  += [str(p) for p in img_dir.glob("*.png") if p.exists()]
            all_samples_by_class.setdefault(class_to_idx[cls], []).extend(paths)

        if indian_dir:
            indian_path = Path(indian_dir)
            if indian_path.exists():
                for dish_dir in sorted(indian_path.iterdir()):
                    if not dish_dir.is_dir():
                        continue
                    dish_name = dish_dir.name.lower().replace(" ", "_")
                    if dish_name not in class_to_idx:
                        class_to_idx[dish_name] = len(self.classes)
                        self.classes.append(dish_name)
                    idx   = class_to_idx[dish_name]
                    paths = ([str(p) for p in dish_dir.glob("*.jpg")] +
                             [str(p) for p in dish_dir.glob("*.jpeg")] +
                             [str(p) for p in dish_dir.glob("*.png")])
                    all_samples_by_class.setdefault(idx, []).extend(
                        [p for p in paths if Path(p).exists()])
            else:
                print(f"[WARN] INDIAN_DIR not found: {indian_path}. Using Food-101 only.")

        self.num_classes = len(self.classes)
        print(f"Total classes: {self.num_classes}")

        self._total_counts = {idx: len(ps) for idx, ps in all_samples_by_class.items()}

        self.minority_classes = {
            idx for idx, ps in all_samples_by_class.items()
            if len(ps) < minority_threshold
        }
        print(f"Minority classes (< {minority_threshold} imgs): {len(self.minority_classes)}")

        rng = np.random.default_rng(seed)
        train_samples, val_samples = [], []

        for idx in sorted(all_samples_by_class.keys()):
            paths = list(all_samples_by_class[idx])
            rng.shuffle(paths)

            if len(paths) == 0:
                continue
            elif len(paths) == 1:
                train_samples.append((paths[0], idx))
            else:
                n_train = max(1, int(round(len(paths) * train_ratio)))
                n_train = min(n_train, len(paths) - 1)
                for p in paths[:n_train]:
                    train_samples.append((p, idx))
                for p in paths[n_train:]:
                    val_samples.append((p, idx))

        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        self.samples = train_samples if split == "train" else val_samples

        self.majority_transform = _base_train_aug(IMAGE_SIZE)
        self.minority_transform = _minority_train_aug(IMAGE_SIZE)
        self.val_transform      = _val_transform(IMAGE_SIZE)

        print(f"[{split.upper()}] {len(self.samples)} samples, {self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.split == "train":
            tfm = self.minority_transform if label in self.minority_classes else self.majority_transform
        else:
            tfm = self.val_transform
        return tfm(image), label

    def make_weighted_sampler(self, boost=MINORITY_BOOST):
        total = sum(self._total_counts.values())
        class_weight = {
            idx: total / max(count, 1)
            for idx, count in self._total_counts.items()
        }
        for idx in self.minority_classes:
            if idx in class_weight:
                class_weight[idx] *= boost

        sample_weights = [class_weight[label] for _, label in self.samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


# ─────────────────────────────────────────────
#  MIXUP / CUTMIX
# ─────────────────────────────────────────────

def mixup_data(x, y, alpha=0.4):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    _, _, H, W = x.shape
    index = torch.randperm(x.size(0), device=x.device)
    cut_w = int(W * np.sqrt(1.0 - lam))
    cut_h = int(H * np.sqrt(1.0 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────

def build_model(model_name, num_classes, pretrained=True):
    weights = "DEFAULT" if pretrained else None

    if model_name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=weights)
        m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.classifier[1].in_features, num_classes))
    elif model_name == "resnet50":
        m = models.resnet50(weights=weights)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, num_classes))
    elif model_name == "vit":
        m = models.vit_b_16(weights=weights)
        m.heads.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.heads.head.in_features, num_classes))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"\nModel: {model_name.upper()}  |  Params: {total:,}  |  Trainable: {trainable:,}\n")
    return m


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, use_mixup=True):
    model.train()
    running_loss = correct = total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        mixed = False
        if use_mixup and random.random() < 0.5:
            if random.random() < 0.5:
                images, ta, tb, lam = mixup_data(images, labels)
            else:
                images, ta, tb, lam = cutmix_data(images, labels)
            mixed = True

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            out  = model(images)
            loss = mixup_criterion(criterion, out, ta, tb, lam) if mixed else criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, pred  = out.max(1)
        correct += pred.eq(labels).sum().item()
        total   += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.*correct/total:.1f}%")

    return running_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = correct_top1 = correct_top5 = total = 0

    for images, labels in tqdm(loader, desc="  [Val]", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            out  = model(images)
            loss = criterion(out, labels)

        running_loss  += loss.item() * images.size(0)
        _, p1          = out.max(1)
        correct_top1  += p1.eq(labels).sum().item()
        _, p5          = out.topk(min(5, out.size(1)), dim=1)
        correct_top5  += p5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
        total         += labels.size(0)

    return running_loss / total, 100.*correct_top1/total, 100.*correct_top5/total


# ─────────────────────────────────────────────
#  PLOTTING + ARTIFACTS
# ─────────────────────────────────────────────

def save_training_plots(history, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Food Training Results", fontsize=14, fontweight="bold")
    ep = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], label="Train", color="#2563eb")
    axes[0].plot(ep, history["val_loss"],   label="Val",   color="#dc2626")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, history["train_acc"],  label="Train", color="#2563eb")
    axes[1].plot(ep, history["val_acc"],    label="Val",   color="#dc2626")
    axes[1].set_title("Top-1 Accuracy (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(ep, history["val_top5"],   label="Val Top-5", color="#16a34a")
    axes[2].set_title("Top-5 Accuracy (%)"); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    out = Path(save_dir) / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curves → {out}")


def save_utils_artifacts(utils_dir, ckpt_path, history, classes, model_name, best_epoch, data_stats,
                          minority_classes, minority_threshold, minority_boost):
    utils_dir = Path(utils_dir)
    utils_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt_path, utils_dir / "best_model.pth")
    bi = best_epoch - 1
    json.dump({
        "model_name": model_name, "best_epoch": best_epoch,
        "num_classes": len(classes),
        "train_samples": data_stats["train_samples"], "test_samples": data_stats["test_samples"],
        "train_ratio": data_stats["train_ratio"],     "test_ratio":   data_stats["test_ratio"],
        "best_train_loss": history["train_loss"][bi], "best_train_accuracy": history["train_acc"][bi],
        "best_test_loss":  history["val_loss"][bi],
        "best_test_top1_accuracy": history["val_acc"][bi],
        "best_test_top5_accuracy": history["val_top5"][bi],
        "final_test_top1_accuracy": history["val_acc"][-1],
        "final_test_top5_accuracy": history["val_top5"][-1],
    }, open(utils_dir / "model_stats.json", "w"), indent=2)
    json.dump(history, open(utils_dir / "training_history.json", "w"), indent=2)
    json.dump(classes, open(utils_dir / "classes.json", "w"), indent=2)

    aug_config = {
        "minority_threshold": minority_threshold,
        "minority_boost": minority_boost,
        "minority_class_indices": sorted(minority_classes),
        "minority_class_names": sorted([classes[i] for i in minority_classes]),
        "num_minority_classes": len(minority_classes),
        "majority_augmentation": [
            "RandomResizedCrop(scale=0.75-1.0)", "RandomHorizontalFlip(0.5)",
            "RandAugment(num_ops=2, magnitude=7)", "RandomRotation(15)",
            "ColorJitter(0.25,0.25,0.25,0.08)", "RandomErasing(p=0.25)",
        ],
        "minority_augmentation": [
            "RandomResizedCrop(scale=0.60-1.0)", "RandomHorizontalFlip(0.5)",
            "RandomVerticalFlip(0.15)", "TrivialAugmentWide()", "RandomRotation(30)",
            "ColorJitter(0.35,0.35,0.35,0.12)", "RandomGrayscale(0.05)", "RandomErasing(p=0.35)",
        ],
        "mixup_cutmix": "50% chance per batch after warmup epoch, alternating MixUp(a=0.4)/CutMix(a=1.0)",
    }
    json.dump(aug_config, open(utils_dir / "augmentation_config.json", "w"), indent=2)

    print(f"Artifacts → {utils_dir}")
    print(f"Augmentation config saved → {utils_dir / 'augmentation_config.json'}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Food Training  |  {MODEL.upper()}  |  {EPOCHS} epochs  |  {DEVICE}")
    print(f"{'='*60}\n")

    train_ds = CombinedFoodDataset(FOOD101_DIR, INDIAN_DIR, split="train")
    val_ds   = CombinedFoodDataset(FOOD101_DIR, INDIAN_DIR, split="val")
    assert train_ds.num_classes == val_ds.num_classes, "Class list mismatch between splits!"
    NUM_CLASSES = train_ds.num_classes

    sampler = train_ds.make_weighted_sampler()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"), drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=="cuda"))

    total_samples = len(train_ds) + len(val_ds)
    data_stats = {
        "train_samples": len(train_ds), "test_samples": len(val_ds),
        "train_ratio": round(len(train_ds)/total_samples, 4),
        "test_ratio":  round(len(val_ds)/total_samples,   4),
    }

    model = build_model(MODEL, NUM_CLASSES, pretrained=True).to(DEVICE)

    head_params = {"efficientnet_b3": "classifier", "resnet50": "fc", "vit": "heads"}
    head_key = head_params[MODEL]
    for name, param in model.named_parameters():
        param.requires_grad = (head_key in name)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.01)
    scaler    = torch.amp.GradScaler(device=DEVICE.type, enabled=(DEVICE.type == "cuda"))

    UNFREEZE_EPOCH = 3
    scheduler      = None
    save_dir       = Path(SAVE_DIR) / MODEL
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_epoch   = 0
    best_wts     = copy.deepcopy(model.state_dict())
    history      = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_top5"]}

    for epoch in range(EPOCHS):
        if epoch == UNFREEZE_EPOCH:
            print(f"\n[Epoch {epoch+1}] Unfreezing ALL layers...\n")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(EPOCHS - UNFREEZE_EPOCH), eta_min=1e-6)

        t0 = time.time()
        train_loss, train_acc         = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, use_mixup=(epoch >= UNFREEZE_EPOCH))
        val_loss,   val_acc, val_top5 = evaluate(model, val_loader, criterion)
        if scheduler: scheduler.step()

        print(f"Epoch {epoch+1:>3}/{EPOCHS}  "
              f"TrainLoss: {train_loss:.4f}  TrainAcc: {train_acc:.2f}%  "
              f"ValLoss: {val_loss:.4f}  ValAcc: {val_acc:.2f}%  "
              f"Top5: {val_top5:.2f}%  LR: {optimizer.param_groups[0]['lr']:.2e}  "
              f"Time: {time.time()-t0:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            best_wts     = copy.deepcopy(model.state_dict())
            ckpt_path    = save_dir / "best_model.pth"
            torch.save({
                "epoch": epoch+1, "model_name": MODEL, "state_dict": best_wts,
                "val_acc": val_acc, "val_top5": val_top5, "image_size": IMAGE_SIZE,
                "train_samples": len(train_ds), "test_samples": len(val_ds),
                "train_ratio": data_stats["train_ratio"], "test_ratio": data_stats["test_ratio"],
                "classes": train_ds.classes,
                "minority_classes": list(train_ds.minority_classes),
            }, ckpt_path)
            print(f"  ★ New best {val_acc:.2f}% → {ckpt_path}")

        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss);     history["val_acc"].append(val_acc)
        history["val_top5"].append(val_top5)

    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), save_dir / "final_weights.pth")
    json.dump(history, open(save_dir / "history.json", "w"), indent=2)
    save_training_plots(history, str(save_dir))
    save_utils_artifacts(UTILS_DIR, save_dir / "best_model.pth", history,
                         train_ds.classes, MODEL, best_epoch, data_stats,
                         train_ds.minority_classes, MINORITY_THRESHOLD, MINORITY_BOOST)

    print(f"\n{'='*60}")
    print(f"  Done! Best Val Top-1: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()