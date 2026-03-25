"""Day 23: Skin Lesion Classification — Data Pipeline"""
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


class SkinDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images, self.labels, self.transform = images, labels, transform
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def get_transforms():
    """ImageNet-normalized transforms (required for pretrained ResNet)."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def load_data():
    logger.info("=" * 60)
    logger.info("LOADING SKIN LESION DATASET")
    logger.info("=" * 60)

    real_path = os.path.join(config.DATA_DIR, "skin_lesions")
    if os.path.exists(real_path):
        return _load_real(real_path)

    logger.info("Generating synthetic skin lesion images (7 classes)...")
    return _generate_synthetic()


def _load_real(path):
    images, labels = [], []
    for ci, cn in enumerate(config.CLASS_NAMES):
        d = os.path.join(path, cn)
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = np.array(Image.open(os.path.join(d, f)).convert('RGB').resize(
                        (config.IMG_SIZE, config.IMG_SIZE)))
                    images.append(img); labels.append(ci)
                except: continue
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int8)


def _generate_synthetic():
    rng = np.random.RandomState(config.RANDOM_SEED)
    s = config.IMG_SIZE
    n_per = config.SYNTHETIC_N // config.NUM_CLASSES
    images, labels = [], []

    # Color profiles per class (RGB base colors for skin lesions)
    profiles = {
        0: {"bg": (200, 170, 140), "lesion": (120, 80, 60), "var": 20},    # Nevi (brown)
        1: {"bg": (190, 160, 135), "lesion": (40, 20, 30), "var": 30},     # Melanoma (dark irregular)
        2: {"bg": (210, 180, 155), "lesion": (160, 130, 90), "var": 15},   # Benign Keratosis (tan)
        3: {"bg": (200, 165, 140), "lesion": (180, 100, 100), "var": 25},  # Basal Cell (pearly/pink)
        4: {"bg": (205, 175, 150), "lesion": (170, 110, 80), "var": 18},   # Actinic Keratosis (rough)
        5: {"bg": (195, 160, 140), "lesion": (150, 40, 50), "var": 22},    # Vascular (red)
        6: {"bg": (200, 170, 145), "lesion": (140, 100, 80), "var": 12},   # Dermatofibroma (firm brown)
    }

    for cls in range(config.NUM_CLASSES):
        p = profiles[cls]
        for _ in range(n_per):
            # Skin background with texture
            img = np.zeros((s, s, 3), dtype=np.float32)
            for c in range(3):
                img[:, :, c] = rng.normal(p["bg"][c], 15, (s, s))

            # Skin texture (fine noise)
            img += rng.normal(0, 5, (s, s, 3))

            # Lesion shape (irregular for melanoma, rounder for others)
            cx, cy = s // 2 + rng.randint(-15, 16), s // 2 + rng.randint(-15, 16)
            if cls == 1:  # melanoma: irregular boundary
                r_base = rng.randint(25, 50)
                angles = np.linspace(0, 2 * np.pi, 36)
                radii = r_base + rng.normal(0, r_base * 0.3, 36)
            else:  # rounder for other types
                r_base = rng.randint(20, 45)
                angles = np.linspace(0, 2 * np.pi, 36)
                radii = r_base + rng.normal(0, r_base * 0.1, 36)

            yy, xx = np.ogrid[:s, :s]
            # Approximate lesion mask using distance from center with angular variation
            angle_map = np.arctan2(yy - cy, xx - cx) % (2 * np.pi)
            r_interp = np.interp(angle_map.ravel(), angles, radii).reshape(s, s)
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            lesion_mask = dist < r_interp

            # Fill lesion with class-specific color + variation
            for c in range(3):
                lesion_color = rng.normal(p["lesion"][c], p["var"], lesion_mask.sum())
                img[:, :, c][lesion_mask] = lesion_color

            # Melanoma-specific: asymmetric color patches within lesion
            if cls == 1:
                n_patches = rng.randint(2, 5)
                for _ in range(n_patches):
                    px, py = cx + rng.randint(-20, 21), cy + rng.randint(-20, 21)
                    pr = rng.randint(5, 15)
                    patch = ((xx - px) ** 2 + (yy - py) ** 2) < pr ** 2
                    patch &= lesion_mask
                    colors=[(60, 30, 40), (100, 60, 50), (30, 10, 20)]
                    idx = rng.choice(len(colors))
                    patch_color = colors[idx]
                    for c in range(3):
                        img[:, :, c][patch] = rng.normal(patch_color[c], 10, patch.sum())

            # Vascular: red streaks
            if cls == 5:
                for _ in range(rng.randint(3, 7)):
                    sx, sy = cx + rng.randint(-20, 21), cy + rng.randint(-20, 21)
                    for step in range(rng.randint(10, 25)):
                        nx = int(sx + step * rng.normal(0.5, 0.3))
                        ny = int(sy + step * rng.normal(0, 0.5))
                        if 0 <= nx < s and 0 <= ny < s and lesion_mask[ny, nx]:
                            img[max(0,ny-1):ny+2, max(0,nx-1):nx+2, 0] = rng.normal(200, 15)
                            img[max(0,ny-1):ny+2, max(0,nx-1):nx+2, 1] = rng.normal(40, 10)

            images.append(np.clip(img, 0, 255).astype(np.uint8))
            labels.append(cls)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int8)
    idx = rng.permutation(len(labels))
    images, labels = images[idx], labels[idx]

    logger.info(f"Generated {len(images)} skin lesion images ({s}x{s} RGB)")
    for i, name in enumerate(config.CLASS_NAMES):
        logger.info(f"  {name:25s}: {(labels == i).sum()}")
    return images, labels


def create_loaders(images, labels):
    logger.info("-" * 60)
    logger.info("CREATING DATALOADERS")
    train_tf, val_tf = get_transforms()
    n = len(labels)
    n_test = int(n * 0.15); n_val = int(n * 0.15); n_train = n - n_val - n_test

    torch.manual_seed(config.RANDOM_SEED)
    idx = torch.randperm(n).tolist()

    train_ds = SkinDataset(images[idx[:n_train]], labels[idx[:n_train]], train_tf)
    val_ds = SkinDataset(images[idx[n_train:n_train + n_val]], labels[idx[n_train:n_train + n_val]], val_tf)
    test_ds = SkinDataset(images[idx[n_train + n_val:]], labels[idx[n_train + n_val:]], val_tf)

    pin = config.DEVICE == "cuda"
    kw = dict(num_workers=0, pin_memory=pin)
    train_ld = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, **kw)
    val_ld = DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False, **kw)
    test_ld = DataLoader(test_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False, **kw)

    logger.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ld, val_ld, test_ld


def plot_samples(images, labels):
    fig, axes = plt.subplots(config.NUM_CLASSES, 4, figsize=(12, 3 * config.NUM_CLASSES))
    for row, (cls, name) in enumerate(enumerate(config.CLASS_NAMES)):
        idxs = np.where(labels == cls)[0][:4]
        for col, i in enumerate(idxs):
            axes[row, col].imshow(images[i])
            axes[row, col].axis('off')
        axes[row, 0].set_ylabel(name[:15], fontsize=9, fontweight='bold')
    plt.suptitle("Skin Lesions -- 7 Classes (ISIC-style)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_samples.png")
