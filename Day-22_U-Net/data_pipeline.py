"""Day 22: Brain Tumor Segmentation — Data Pipeline"""
import logging, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import config

logger = logging.getLogger(__name__)


class BrainMRIDataset(Dataset):
    def __init__(self, images, masks):
        # images: (N, H, W) uint8, masks: (N, H, W) uint8 (0 or 1)
        self.images = images.astype(np.float32) / 255.0
        self.masks = masks.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)   # (1, H, W)
        mask = torch.from_numpy(self.masks[idx]).unsqueeze(0)   # (1, H, W)
        return img, mask


def load_data():
    logger.info("=" * 60)
    logger.info("LOADING BRAIN MRI DATASET")
    logger.info("=" * 60)

    rng = np.random.RandomState(config.RANDOM_SEED)
    n = config.SYNTHETIC_N
    s = config.IMG_SIZE
    images = np.zeros((n, s, s), dtype=np.uint8)
    masks = np.zeros((n, s, s), dtype=np.uint8)

    for i in range(n):
        img = np.zeros((s, s), dtype=np.float32)
        mask = np.zeros((s, s), dtype=np.uint8)

        # Brain ellipse (gray matter)
        cy, cx = s // 2 + rng.randint(-3, 4), s // 2 + rng.randint(-3, 4)
        ry, rx = s // 2 - 10 + rng.randint(-5, 6), s // 2 - 15 + rng.randint(-5, 6)
        yy, xx = np.ogrid[:s, :s]
        brain_mask = ((xx - cx) ** 2 / max(rx, 1) ** 2 + (yy - cy) ** 2 / max(ry, 1) ** 2) < 1

        # Brain tissue with texture
        brain_intensity = rng.normal(140, 20, (s, s)).clip(80, 200)
        # Add ventricle-like darker center region
        vent_mask = ((xx - cx) ** 2 / max(rx // 4, 1) ** 2 + (yy - cy) ** 2 / max(ry // 4, 1) ** 2) < 1
        brain_intensity[vent_mask] = rng.normal(60, 15, vent_mask.sum()).clip(30, 90)

        img[brain_mask] = brain_intensity[brain_mask]
        # Skull ring (bright boundary)
        skull = brain_mask & ~(((xx - cx) ** 2 / max(rx - 4, 1) ** 2 + (yy - cy) ** 2 / max(ry - 4, 1) ** 2) < 1)
        img[skull] = rng.normal(200, 15, skull.sum()).clip(170, 255)

        # Background noise
        img[~brain_mask] = rng.normal(10, 5, (~brain_mask).sum()).clip(0, 30)

        # Tumor (~65% of slices have tumors, rest are healthy)
        has_tumor = rng.random() < 0.65
        if has_tumor:
            # Random tumor position inside brain
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.15, 0.55) * min(rx, ry)
            tx = int(cx + dist * np.cos(angle))
            ty = int(cy + dist * np.sin(angle))
            tr = rng.randint(6, 22)  # tumor radius

            tumor_region = ((xx - tx) ** 2 + (yy - ty) ** 2) <= tr ** 2
            tumor_region &= brain_mask  # keep inside brain

            if tumor_region.sum() > 20:  # valid tumor
                # Tumor: brighter heterogeneous mass
                img[tumor_region] = rng.normal(190, 30, tumor_region.sum()).clip(140, 255)
                # Necrotic core (darker center)
                core = ((xx - tx) ** 2 + (yy - ty) ** 2) <= (tr // 3) ** 2
                core &= brain_mask
                img[core] = rng.normal(80, 20, core.sum()).clip(40, 120)
                mask[tumor_region] = 1

        # Add gaussian noise
        img += rng.normal(0, 5, (s, s))
        images[i] = np.clip(img, 0, 255).astype(np.uint8)
        masks[i] = mask

    n_tumor = (masks.sum(axis=(1, 2)) > 0).sum()
    logger.info(f"Generated {n} MRI slices ({s}x{s})")
    logger.info(f"  With tumor: {n_tumor} ({n_tumor/n*100:.1f}%) | Without: {n - n_tumor}")
    logger.info(f"  Avg tumor pixels per positive slice: {masks[masks.sum(axis=(1,2))>0].sum(axis=(1,2)).mean():.0f}")
    return images, masks


def explore_data(images, masks):
    logger.info("-" * 60)
    logger.info("EDA")

    fig, axes = plt.subplots(3, 6, figsize=(20, 10))

    # Row 0: samples with tumors
    tumor_idx = np.where(masks.sum(axis=(1, 2)) > 0)[0]
    rng = np.random.RandomState(config.RANDOM_SEED)
    samples = rng.choice(tumor_idx, min(6, len(tumor_idx)), replace=False)
    for col, idx in enumerate(samples):
        axes[0, col].imshow(images[idx], cmap='gray')
        axes[0, col].set_title(f"MRI #{idx}", fontsize=9)
        axes[0, col].axis('off')
    axes[0, 0].set_ylabel("MRI Image", fontsize=11, fontweight='bold')

    # Row 1: corresponding masks
    for col, idx in enumerate(samples):
        axes[1, col].imshow(masks[idx], cmap='Reds', vmin=0, vmax=1)
        axes[1, col].set_title(f"Tumor pixels: {masks[idx].sum()}", fontsize=9)
        axes[1, col].axis('off')
    axes[1, 0].set_ylabel("Ground Truth", fontsize=11, fontweight='bold')

    # Row 2: overlay
    for col, idx in enumerate(samples):
        overlay = np.stack([images[idx]] * 3, axis=-1).astype(np.float32) / 255.0
        overlay[masks[idx] == 1, 0] = 1.0  # red channel for tumor
        overlay[masks[idx] == 1, 1] *= 0.3
        overlay[masks[idx] == 1, 2] *= 0.3
        axes[2, col].imshow(overlay)
        axes[2, col].axis('off')
    axes[2, 0].set_ylabel("Overlay", fontsize=11, fontweight='bold')

    plt.suptitle("Brain MRI -- Tumor Segmentation Ground Truth", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_samples.png")

    # Tumor size distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tumor_sizes = masks.sum(axis=(1, 2))
    axes[0].hist(tumor_sizes[tumor_sizes > 0], bins=30, color='#EF5350', edgecolor='white', alpha=0.85)
    axes[0].set_title("Tumor Size Distribution (pixels)", fontweight='bold')
    axes[0].set_xlabel("Tumor area (pixels)"); axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].bar(["Tumor", "No Tumor"], [(tumor_sizes > 0).sum(), (tumor_sizes == 0).sum()],
                color=['#EF5350', '#66BB6A'], edgecolor='white')
    axes[1].set_title("Slice Distribution", fontweight='bold')
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_tumor_stats.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_tumor_stats.png")


def create_loaders(images, masks):
    logger.info("-" * 60)
    logger.info("CREATING DATALOADERS")

    n = len(images)
    n_test = int(n * 0.15); n_val = int(n * 0.15); n_train = n - n_val - n_test

    torch.manual_seed(config.RANDOM_SEED)
    idx = torch.randperm(n).tolist()

    train_ds = BrainMRIDataset(images[idx[:n_train]], masks[idx[:n_train]])
    val_ds = BrainMRIDataset(images[idx[n_train:n_train + n_val]], masks[idx[n_train:n_train + n_val]])
    test_ds = BrainMRIDataset(images[idx[n_train + n_val:]], masks[idx[n_train + n_val:]])

    pin = config.DEVICE == "cuda"
    train_ld = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    test_ld = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

    logger.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ld, val_ld, test_ld
