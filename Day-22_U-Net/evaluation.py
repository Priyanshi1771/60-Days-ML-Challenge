"""Day 22: Brain Tumor Segmentation — Evaluation"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast
import config

logger = logging.getLogger(__name__)


def dice_score(pred, target, smooth=1.0):
    p, t = pred.ravel(), target.ravel()
    inter = (p * t).sum()
    return (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)


def iou_score(pred, target, smooth=1.0):
    p, t = pred.ravel(), target.ravel()
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return (inter + smooth) / (union + smooth)


@torch.no_grad()
def evaluate_model(model, test_loader):
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)

    model.eval()
    all_dice, all_iou = [], []
    all_imgs, all_masks, all_preds = [], [], []

    for imgs, masks in test_loader:
        imgs_gpu = imgs.to(config.DEVICE, non_blocking=True)
        with autocast(device_type=config.DEVICE, enabled=config.DEVICE == "cuda"):
            logits = model(imgs_gpu)
        preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
        masks_np = masks.numpy()
        imgs_np = imgs.numpy()

        for j in range(len(preds)):
            p = preds[j, 0]; m = masks_np[j, 0]
            all_dice.append(dice_score(p, m))
            all_iou.append(iou_score(p, m))

        all_imgs.append(imgs_np)
        all_masks.append(masks_np)
        all_preds.append(preds)

    all_imgs = np.concatenate(all_imgs)
    all_masks = np.concatenate(all_masks)
    all_preds = np.concatenate(all_preds)
    all_dice = np.array(all_dice)
    all_iou = np.array(all_iou)

    # Only evaluate on slices that actually have tumors
    has_tumor = all_masks[:, 0].sum(axis=(1, 2)) > 0
    logger.info(f"  Overall Dice:  {all_dice.mean():.4f} +/- {all_dice.std():.4f}")
    logger.info(f"  Overall IoU:   {all_iou.mean():.4f} +/- {all_iou.std():.4f}")
    logger.info(f"  Tumor slices:  Dice={all_dice[has_tumor].mean():.4f} | IoU={all_iou[has_tumor].mean():.4f}")
    logger.info(f"  Normal slices: Dice={all_dice[~has_tumor].mean():.4f} (should be ~1.0)")

    _plot_predictions(all_imgs, all_masks, all_preds, all_dice, has_tumor)
    _plot_metrics(all_dice, all_iou, has_tumor)

    results = {
        "Dice (all)": all_dice.mean(), "IoU (all)": all_iou.mean(),
        "Dice (tumor only)": all_dice[has_tumor].mean(),
        "IoU (tumor only)": all_iou[has_tumor].mean(),
    }
    return results


def _plot_predictions(imgs, masks, preds, dices, has_tumor):
    """Show best and worst predictions side by side."""
    tumor_idx = np.where(has_tumor)[0]
    if len(tumor_idx) < 6:
        sample_idx = tumor_idx
    else:
        sorted_by_dice = tumor_idx[np.argsort(dices[tumor_idx])]
        # 3 worst + 3 best
        sample_idx = np.concatenate([sorted_by_dice[:3], sorted_by_dice[-3:]])

    n = len(sample_idx)
    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 10))
    if n == 1:
        axes = axes.reshape(3, 1)

    for col, idx in enumerate(sample_idx):
        img = imgs[idx, 0]
        gt = masks[idx, 0]
        pred = preds[idx, 0]

        axes[0, col].imshow(img, cmap='gray'); axes[0, col].axis('off')
        axes[0, col].set_title(f"Dice={dices[idx]:.3f}", fontsize=9)

        # Overlay: green=correct, red=false positive, blue=false negative
        overlay = np.stack([img] * 3, axis=-1)
        overlay = overlay / overlay.max() if overlay.max() > 0 else overlay
        tp = (pred == 1) & (gt == 1)
        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)
        overlay[tp, 1] = 1.0; overlay[tp, 0] *= 0.3; overlay[tp, 2] *= 0.3  # green
        overlay[fp, 0] = 1.0; overlay[fp, 1] *= 0.3; overlay[fp, 2] *= 0.3  # red
        overlay[fn, 2] = 1.0; overlay[fn, 0] *= 0.3; overlay[fn, 1] *= 0.3  # blue
        axes[1, col].imshow(overlay); axes[1, col].axis('off')

        axes[2, col].imshow(pred, cmap='Reds', vmin=0, vmax=1); axes[2, col].axis('off')

    axes[0, 0].set_ylabel("MRI", fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel("Overlay\nG=TP R=FP B=FN", fontsize=9, fontweight='bold')
    axes[2, 0].set_ylabel("Prediction", fontsize=11, fontweight='bold')

    plt.suptitle("U-Net Predictions -- Worst (left) to Best (right)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_predictions.png")


def _plot_metrics(dices, ious, has_tumor):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Dice distribution
    axes[0].hist(dices[has_tumor], bins=25, color='#7E57C2', edgecolor='white', alpha=0.85, label='Tumor slices')
    axes[0].hist(dices[~has_tumor], bins=10, color='#66BB6A', edgecolor='white', alpha=0.6, label='Normal slices')
    axes[0].set_title("Dice Score Distribution", fontweight='bold')
    axes[0].set_xlabel("Dice"); axes[0].legend(); axes[0].spines[['top', 'right']].set_visible(False)

    # IoU distribution
    axes[1].hist(ious[has_tumor], bins=25, color='#FF7043', edgecolor='white', alpha=0.85)
    axes[1].set_title("IoU Distribution (tumor slices)", fontweight='bold')
    axes[1].set_xlabel("IoU"); axes[1].spines[['top', 'right']].set_visible(False)

    # Summary bar
    metrics = {
        "Dice\n(all)": dices.mean(),
        "Dice\n(tumor)": dices[has_tumor].mean(),
        "IoU\n(tumor)": ious[has_tumor].mean(),
    }
    colors = ['#7E57C2', '#4FC3F7', '#FF7043']
    bars = axes[2].bar(metrics.keys(), metrics.values(), color=colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars, metrics.values()):
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
    axes[2].set_ylim(0, 1.05); axes[2].set_title("Summary Metrics", fontweight='bold')
    axes[2].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_metrics.png")


def save_results(results):
    pd.DataFrame([results]).to_csv(f"{config.OUTPUT_DIR}/day22_results.csv", index=False, float_format='%.4f')
    with open(f"{config.OUTPUT_DIR}/day22_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 22: BRAIN TUMOR SEGMENTATION -- U-NET\n" + "=" * 70 + "\n\n")
        f.write("FIRST SEGMENTATION PROJECT!\n\n")
        for k, v in results.items():
            f.write(f"  {k:25s}: {v:.4f}\n")
        f.write("\nKEY TAKEAWAYS:\n")
        f.write("1. U-Net's skip connections preserve spatial detail lost during downsampling\n")
        f.write("2. Dice loss handles class imbalance (tiny tumor vs large background)\n")
        f.write("3. Dice score > 0.7 is clinically useful for tumor segmentation\n")
        f.write("4. False negatives (missed tumor) are worse than false positives\n")
        f.write("5. The encoder-decoder architecture generalizes to many medical imaging tasks\n")
    logger.info("Results saved")
