"""Day 21: Pneumonia Detection — Data Pipeline"""
import logging, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config

logger = logging.getLogger(__name__)


class XrayDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images, self.labels, self.transform = images, labels, transform
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx]).convert('L')
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return train_tf, val_tf


def load_data():
    logger.info("=" * 60)
    logger.info("LOADING CHEST X-RAY DATASET")
    logger.info("=" * 60)

    real_path = os.path.join(config.DATA_DIR, "chest_xray")
    if os.path.exists(real_path):
        return _load_real(real_path)

    logger.info("Generating synthetic chest X-rays...")
    return _generate_synthetic()


def _load_real(path):
    images, labels = [], []
    for split in ["train", "val", "test"]:
        for ci, cn in enumerate(config.CLASS_NAMES):
            d = os.path.join(path, split, cn)
            if not os.path.exists(d): continue
            for f in os.listdir(d):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img = np.array(Image.open(os.path.join(d, f)).convert('L').resize(
                            (config.IMG_SIZE, config.IMG_SIZE)))
                        images.append(img); labels.append(ci)
                    except: continue
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int8)


def _generate_synthetic():
    rng = np.random.RandomState(config.RANDOM_SEED)
    s = config.IMG_SIZE
    n_per = config.SYNTHETIC_N // 2
    images, labels = [], []

    for _ in range(n_per):
        # NORMAL: clear lungs + rib lines
        img = rng.normal(40, 10, (s, s)).clip(0, 255)
        cy, cx = s // 2, s // 2
        for dx in [-s // 4, s // 4]:
            yy, xx = np.ogrid[:s, :s]
            lung = ((xx - cx - dx) ** 2 / (s // 4) ** 2 + (yy - cy) ** 2 / (s // 3) ** 2) < 1
            img[lung] = rng.normal(25, 8, lung.sum()).clip(5, 60)
        for ry in range(s // 6, s - s // 6, s // 8):
            img[ry:ry + 2, s // 4:3 * s // 4] = rng.normal(70, 10, (2, s // 2)).clip(40, 100)
        images.append(img.astype(np.uint8)); labels.append(0)

    for _ in range(n_per):
        # PNEUMONIA: hazy white opacity patches
        img = rng.normal(50, 12, (s, s)).clip(0, 255)
        cy, cx = s // 2, s // 2
        for dx in [-s // 4, s // 4]:
            yy, xx = np.ogrid[:s, :s]
            lung = ((xx - cx - dx) ** 2 / (s // 4) ** 2 + (yy - cy) ** 2 / (s // 3) ** 2) < 1
            img[lung] = rng.normal(35, 10, lung.sum()).clip(10, 70)
        for _ in range(rng.randint(2, 6)):
            py, px = rng.randint(s // 4, 3 * s // 4), rng.randint(s // 4, 3 * s // 4)
            pr = rng.randint(8, 25)
            yy, xx = np.ogrid[-pr:pr + 1, -pr:pr + 1]
            mask = xx ** 2 + yy ** 2 <= pr ** 2
            y1, y2 = max(0, py - pr), min(s, py + pr + 1)
            x1, x2 = max(0, px - pr), min(s, px + pr + 1)
            m = mask[:y2 - y1, :x2 - x1]
            img[y1:y2, x1:x2][m] = rng.normal(120, 25, m.sum()).clip(80, 200)
        images.append(img.astype(np.uint8)); labels.append(1)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int8)
    idx = rng.permutation(len(labels))
    logger.info(f"Generated {len(images)} X-rays | Normal: {(labels == 0).sum()} | Pneumonia: {(labels == 1).sum()}")
    return images[idx], labels[idx]


def create_loaders(images, labels):
    logger.info("-" * 60)
    logger.info("CREATING DATALOADERS")
    train_tf, val_tf = get_transforms()
    n = len(labels)
    n_test = int(n * 0.15); n_val = int(n * 0.15); n_train = n - n_val - n_test

    torch.manual_seed(config.RANDOM_SEED)
    idx = torch.randperm(n).tolist()
    train_ds = XrayDataset(images[idx[:n_train]], labels[idx[:n_train]], train_tf)
    val_ds = XrayDataset(images[idx[n_train:n_train + n_val]], labels[idx[n_train:n_train + n_val]], val_tf)
    test_ds = XrayDataset(images[idx[n_train + n_val:]], labels[idx[n_train + n_val:]], val_tf)

    pin = config.DEVICE == "cuda"
    train_ld = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=pin)
    test_ld = DataLoader(test_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=pin)

    logger.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ld, val_ld, test_ld


def plot_samples(images, labels):
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for row, (cls, name) in enumerate([(0, "NORMAL"), (1, "PNEUMONIA")]):
        idxs = np.where(labels == cls)[0][:6]
        for col, i in enumerate(idxs):
            axes[row, col].imshow(images[i], cmap='gray'); axes[row, col].axis('off')
        axes[row, 0].set_ylabel(name, fontsize=12, fontweight='bold')
    plt.suptitle("Chest X-Rays: Normal vs Pneumonia", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_samples.png")
