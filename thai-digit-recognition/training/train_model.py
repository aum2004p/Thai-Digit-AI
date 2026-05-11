"""
Train CNN model for Thai digit recognition (51-55) using PyTorch.
- Strict train/test split (no leakage)
- Weighted sampler for class balance
- Same preprocessing as webapp
"""

import os, sys, json, random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

print(f"PyTorch version: {torch.__version__}")

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE     = 32
NUM_CLASSES  = 5
BATCH_SIZE   = 16
EPOCHS       = 80
LR           = 1e-3
LABELS       = [51, 52, 53, 54, 55]
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(LABELS)}
IDX_TO_LABEL = {i: lbl for i, lbl in enumerate(LABELS)}

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')
MODEL_DIR   = os.path.join(BASE_DIR, '..', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ── Preprocessing (identical to webapp/app.py) ─────────────────────────────────
def preprocess_for_model(img_gray: Image.Image) -> Image.Image:
    arr = np.array(img_gray, dtype=np.uint8)
    if arr.mean() < 128:
        arr = 255 - arr
    binary = (arr < 200).astype(np.uint8)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any():
        return Image.fromarray(np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    pad = max(4, int(max(h, w) * 0.15))
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(arr.shape[1] - 1, cmax + pad)
    cropped = arr[rmin:rmax+1, cmin:cmax+1]
    ch, cw = cropped.shape
    side = max(ch, cw)
    square = np.ones((side, side), dtype=np.uint8) * 255
    top  = (side - ch) // 2
    left = (side - cw) // 2
    square[top:top+ch, left:left+cw] = cropped
    return Image.fromarray(square).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


# ── Dataset ────────────────────────────────────────────────────────────────────
class ThaiDigitDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.transform = transform
        self.samples = []
        split_dir = os.path.join(DATASET_DIR, split)
        for label in LABELS:
            label_dir = os.path.join(split_dir, str(label))
            if not os.path.isdir(label_dir):
                print(f"  WARNING: {label_dir} not found")
                continue
            files = [f for f in sorted(os.listdir(label_dir)) if f.lower().endswith('.png')]
            for fname in files:
                self.samples.append((os.path.join(label_dir, fname), LABEL_TO_IDX[label]))

        # Verify no overlap between train and test by filename
        print(f"  {split}: {len(self.samples)} samples across {NUM_CLASSES} classes")
        counts = [0] * NUM_CLASSES
        for _, idx in self.samples:
            counts[idx] += 1
        for i, c in enumerate(counts):
            print(f"    {LABELS[i]}: {c}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')
        # Apply same preprocessing as webapp
        img = preprocess_for_model(img)
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        return img, label

    def get_class_weights(self):
        """Return per-sample weights for WeightedRandomSampler."""
        counts = [0] * NUM_CLASSES
        for _, idx in self.samples:
            counts[idx] += 1
        weights_per_class = [1.0 / max(c, 1) for c in counts]
        return [weights_per_class[idx] for _, idx in self.samples]


# ── Model ──────────────────────────────────────────────────────────────────────
class ThaiDigitCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Train / eval helpers ───────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if training:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


# ── Main ───────────────────────────────────────────────────────────────────────
def train():
    train_transform = T.Compose([
        T.RandomRotation(15),
        T.RandomAffine(0, scale=(0.85, 1.15)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])

    print("\nLoading datasets...")
    train_ds = ThaiDigitDataset('train', transform=train_transform)
    test_ds  = ThaiDigitDataset('test',  transform=test_transform)

    if len(train_ds) == 0:
        print("ERROR: No training data found! Run generate_dataset.py first.")
        sys.exit(1)

    # Balanced sampler — each class gets equal chance per batch
    sample_weights = train_ds.get_class_weights()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = ThaiDigitCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6)

    best_acc   = 0.0
    best_path  = os.path.join(MODEL_DIR, 'best_model.pt')
    patience_counter = 0
    PATIENCE   = 20
    history    = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, test_loader,  criterion)
        scheduler.step(va_acc)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"{epoch:>6} {tr_loss:>11.4f} {tr_acc*100:>9.2f}% {va_loss:>10.4f} {va_acc*100:>8.2f}% {lr_now:>10.2e}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Load best weights and evaluate
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    final_loss, final_acc = run_epoch(model, test_loader, criterion)
    print(f"\n✅ Best Val Accuracy : {best_acc*100:.2f}%")
    print(f"   Final Val Accuracy: {final_acc*100:.2f}%")

    # Per-class accuracy
    model.eval()
    class_correct = [0] * NUM_CLASSES
    class_total   = [0] * NUM_CLASSES
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            for p, t in zip(preds, labels):
                class_total[t.item()]   += 1
                class_correct[t.item()] += int(p == t)
    print("\nPer-class accuracy:")
    for i in range(NUM_CLASSES):
        n = class_total[i]
        c = class_correct[i]
        print(f"  {LABELS[i]}: {c}/{n} = {100*c/max(n,1):.1f}%")

    # Save
    model_path = os.path.join(MODEL_DIR, 'thai_digit_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': NUM_CLASSES,
        'img_size': IMG_SIZE,
        'labels': LABELS,
        'label_to_idx': LABEL_TO_IDX,
        'idx_to_label': IDX_TO_LABEL,
        'test_accuracy': float(final_acc),
    }, model_path)

    scripted = torch.jit.script(model)
    scripted.save(os.path.join(MODEL_DIR, 'thai_digit_scripted.pt'))

    metadata = {
        'img_size': IMG_SIZE, 'num_classes': NUM_CLASSES,
        'labels': LABELS,
        'label_to_idx': {str(k): v for k, v in LABEL_TO_IDX.items()},
        'idx_to_label': {str(k): v for k, v in IDX_TO_LABEL.items()},
        'test_accuracy': float(final_acc),
        'test_loss': float(final_loss),
        'framework': 'pytorch',
    }
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n🎉 Training complete! Model saved to {model_path}")
    return model


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train()
