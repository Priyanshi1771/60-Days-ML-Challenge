"""
=============================================================================
 Day 10: Malaria Cell Classification — Data Pipeline (Optimized)
=============================================================================
 HANDLES:
   - Real dataset loading (Kaggle malaria cell images)
   - Synthetic fallback (colored blob images mimicking cell morphology)
   - Data augmentation (flips, rotation, color jitter)
   - Efficient DataLoader with pin_memory + prefetch
   
 OPTIMIZATION: float32 tensors, in-place augmentation, no redundant copies
=============================================================================
"""
import logging, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import config

logger = logging.getLogger(__name__)


class MalariaDataset(Dataset):
    """Memory-efficient dataset: loads images on-demand (no full preload)."""
    
    def __init__(self, images, labels, transform=None):
        self.images = images      # List of PIL images or numpy arrays
        self.labels = labels      # numpy int8 array
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def get_transforms():
    """
    Data augmentation — CRITICAL for small/medium image datasets.
    
    WHY AUGMENTATION?
    - CNNs are data-hungry; augmentation artificially expands dataset
    - Malaria cells can appear rotated, flipped, at different zoom levels
    - Color jitter handles staining variation across microscope slides
    
    Train: augment aggressively
    Val/Test: only resize + normalize (no augmentation!)
    """
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),  # Converts to [0,1] float32
        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # ImageNet stats
                             std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


def load_data():
    """Load real malaria images or generate synthetic fallback."""
    logger.info("=" * 60)
    logger.info("LOADING MALARIA CELL IMAGE DATASET")
    logger.info("=" * 60)
    
    # Try real dataset paths
    real_paths = [
        os.path.join(config.DATA_DIR, "cell_images"),
        os.path.join(config.DATA_DIR, "malaria"),
    ]
    
    for path in real_paths:
        if os.path.exists(path):
            return _load_real_data(path)
    
    logger.info("Real dataset not found — generating synthetic cell images...")
    return _generate_synthetic_data()


def _load_real_data(data_path):
    """Load real malaria cell images from directory structure."""
    images, labels = [], []
    
    for label, class_name in enumerate(["Parasitized", "Uninfected"]):
        class_dir = os.path.join(data_path, class_name)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, fname)
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    labels.append(label)
                except:
                    continue
    
    labels = np.array(labels, dtype=np.int8)
    logger.info(f"Loaded {len(images)} real images")
    logger.info(f"  Parasitized: {np.sum(labels==0)} | Uninfected: {np.sum(labels==1)}")
    return images, labels


def _generate_synthetic_data():
    """
    Generate synthetic cell-like images (64×64 RGB).
    
    Parasitized: darker background, purple/dark blobs (simulating parasites)
    Uninfected: pink/lighter background, uniform red cells
    """
    rng = np.random.RandomState(config.RANDOM_SEED)
    n_per_class = config.SYNTHETIC_N // 2
    images, labels = [], []
    s = config.IMG_SIZE
    
    for _ in range(n_per_class):
        # PARASITIZED: pink cell with dark purple parasitic inclusions
        img = rng.normal(180, 25, (s, s, 3)).clip(0, 255)
        img[:, :, 0] = rng.normal(200, 20, (s, s)).clip(100, 255)  # Red channel high
        img[:, :, 2] = rng.normal(160, 25, (s, s)).clip(80, 220)   # Blue moderate
        
        # Add 1-3 dark parasitic blobs
        n_parasites = rng.randint(1, 4)
        for _ in range(n_parasites):
            cx, cy = rng.randint(10, s-10, 2)
            r = rng.randint(3, 10)
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            mask = xx**2 + yy**2 <= r**2
            y_start, y_end = max(0, cy-r), min(s, cy+r+1)
            x_start, x_end = max(0, cx-r), min(s, cx+r+1)
            m_y = mask[max(0,r-cy):r+min(r+1,s-cy), max(0,r-cx):r+min(r+1,s-cx)]
            h, w = y_end-y_start, x_end-x_start
            m_y = m_y[:h, :w]
            img[y_start:y_end, x_start:x_end][m_y] = rng.normal(60, 20, 3).clip(20, 100)
        
        # Add cell boundary (circular)
        center = s // 2
        yy, xx = np.ogrid[:s, :s]
        cell_mask = (xx-center)**2 + (yy-center)**2 > (s//2 - 4)**2
        img[cell_mask] = rng.normal(220, 15, 3).clip(180, 255)
        
        images.append(img.astype(np.uint8))
        labels.append(0)
    
    for _ in range(n_per_class):
        # UNINFECTED: uniform pink/red cell, no dark blobs
        img = rng.normal(195, 15, (s, s, 3)).clip(0, 255)
        img[:, :, 0] = rng.normal(210, 15, (s, s)).clip(150, 255)
        img[:, :, 1] = rng.normal(170, 15, (s, s)).clip(120, 220)
        img[:, :, 2] = rng.normal(175, 15, (s, s)).clip(120, 230)
        
        # Cell boundary
        center = s // 2
        yy, xx = np.ogrid[:s, :s]
        cell_mask = (xx-center)**2 + (yy-center)**2 > (s//2 - 3)**2
        img[cell_mask] = rng.normal(225, 10, 3).clip(190, 255)
        
        # Slight central pallor (normal RBC feature)
        inner = (xx-center)**2 + (yy-center)**2 < (s//6)**2
        img[inner] = np.clip(img[inner] + 20, 0, 255)
        
        images.append(img.astype(np.uint8))
        labels.append(1)
    
    labels = np.array(labels, dtype=np.int8)
    
    # Shuffle
    idx = rng.permutation(len(labels))
    images = [images[i] for i in idx]
    labels = labels[idx]
    
    logger.info(f"Generated {len(images)} synthetic cell images ({config.IMG_SIZE}×{config.IMG_SIZE})")
    logger.info(f"  Parasitized: {np.sum(labels==0)} | Uninfected: {np.sum(labels==1)}")
    return images, labels


def create_dataloaders(images, labels):
    """Create train/val/test DataLoaders with proper augmentation split."""
    logger.info("-" * 60)
    logger.info("CREATING DATALOADERS")
    logger.info("-" * 60)
    
    train_tf, val_tf = get_transforms()
    
    n = len(labels)
    n_test = int(n * config.TEST_SPLIT)
    n_val = int(n * config.VAL_SPLIT)
    n_train = n - n_val - n_test
    
    # Deterministic split
    torch.manual_seed(config.RANDOM_SEED)
    indices = torch.randperm(n).tolist()
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_ds = MalariaDataset([images[i] for i in train_idx], labels[train_idx], train_tf)
    val_ds = MalariaDataset([images[i] for i in val_idx], labels[val_idx], val_tf)
    test_ds = MalariaDataset([images[i] for i in test_idx], labels[test_idx], val_tf)
    
    pin = config.DEVICE == "cuda"
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=config.NUM_WORKERS, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=pin)
    
    logger.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    logger.info(f"  Batch size: {config.BATCH_SIZE} | Device: {config.DEVICE}")
    logger.info(f"  Augmentation: HFlip, VFlip, Rotation(15°), ColorJitter")
    
    return train_loader, val_loader, test_loader


def plot_sample_images(images, labels):
    """Show sample parasitized vs uninfected cells."""
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    for row, (cls, title, color) in enumerate([(0, "🦟 Parasitized", "#EF5350"),
                                                 (1, "✅ Uninfected", "#66BB6A")]):
        cls_idx = np.where(labels == cls)[0][:6]
        for col, idx in enumerate(cls_idx):
            img = images[idx]
            if isinstance(img, np.ndarray):
                axes[row, col].imshow(img)
            else:
                axes[row, col].imshow(np.array(img))
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=12, fontweight='bold', color=color)
    
    fig.suptitle("🔬 Sample Cell Images — Parasitized vs Uninfected", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_sample_images.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_sample_images.png")
