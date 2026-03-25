"""
Day 23: Skin Lesion Classification — Model Training
FIRST TRANSFER LEARNING PROJECT!

Three strategies compared head-to-head:
  1. scratch:  Random weights, train everything from zero
  2. frozen:   Pretrained ResNet, freeze all conv layers, train only FC head
  3. finetune: Pretrained ResNet, unfreeze all layers, train with low LR
"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from torchvision import models
import config

logger = logging.getLogger(__name__)


def build_resnet(strategy="finetune"):
    """Build ResNet18 with specified transfer learning strategy."""
    if strategy == "scratch":
        model = models.resnet18(weights=None)
        lr = config.LR_SCRATCH
    else:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except:
            model = models.resnet18(pretrained=True)

        if strategy == "frozen":
            # Freeze ALL conv layers -- only train the final FC
            for param in model.parameters():
                param.requires_grad = False
            lr = config.LR_FROZEN
        else:  # finetune
            lr = config.LR_FINETUNE

    # Replace final FC for 7-class skin lesion classification
    n_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(n_features, config.NUM_CLASSES)
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  {strategy:10s}: trainable={trainable:>10,} / total={total:>10,} "
                f"({trainable/total*100:.1f}%)")

    return model, lr


def train_one_strategy(model, lr, train_loader, val_loader, name):
    """Train a single model on GPU with AMP."""
    model = model.to(config.DEVICE)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    use_amp = config.DEVICE == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val, patience, best_state = float('inf'), 0, None
    history = {"t_loss": [], "v_loss": [], "t_acc": [], "v_acc": []}

    t0 = time.time()
    for epoch in range(config.EPOCHS):
        # Train
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            lbls = lbls.to(config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=config.DEVICE, enabled=use_amp):
                out = model(imgs); loss = criterion(out, lbls)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            r_loss += loss.item() * imgs.size(0)
            total += lbls.size(0); correct += out.argmax(1).eq(lbls).sum().item()

        t_loss, t_acc = r_loss / total, correct / total

        # Validate
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(config.DEVICE, non_blocking=True)
                lbls = lbls.to(config.DEVICE, non_blocking=True)
                with autocast(device_type=config.DEVICE, enabled=use_amp):
                    out = model(imgs); loss = criterion(out, lbls)
                v_loss += loss.item() * imgs.size(0)
                v_tot += lbls.size(0); v_corr += out.argmax(1).eq(lbls).sum().item()

        vl, va = v_loss / v_tot, v_corr / v_tot
        scheduler.step(vl)

        history["t_loss"].append(t_loss); history["v_loss"].append(vl)
        history["t_acc"].append(t_acc); history["v_acc"].append(va)

        if vl < best_val:
            best_val = vl; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= config.PATIENCE:
            break

    if best_state:
        model.load_state_dict(best_state); model = model.to(config.DEVICE)

    elapsed = time.time() - t0
    best_acc = max(history["v_acc"])
    logger.info(f"  {name:10s}: best_val_acc={best_acc:.4f} | epochs={len(history['t_loss'])} | {elapsed:.1f}s")

    return model, history, best_acc, elapsed


def train_all_strategies(train_loader, val_loader):
    """Train ResNet18 with all 3 transfer learning strategies."""
    logger.info("=" * 60)
    logger.info(f"TRANSFER LEARNING COMPARISON (device={config.DEVICE})")
    logger.info("=" * 60)

    results = {}
    for strategy in config.STRATEGIES:
        logger.info(f"\n--- Strategy: {strategy.upper()} ---")
        model, lr = build_resnet(strategy)
        model, history, best_acc, elapsed = train_one_strategy(
            model, lr, train_loader, val_loader, strategy)
        results[strategy] = {
            "model": model, "history": history,
            "best_acc": best_acc, "time": elapsed
        }

    _plot_strategy_comparison(results)
    return results


def _plot_strategy_comparison(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"scratch": "#EF5350", "frozen": "#4FC3F7", "finetune": "#66BB6A"}

    # Training curves (loss)
    for name, data in results.items():
        ep = range(1, len(data["history"]["t_loss"]) + 1)
        axes[0].plot(ep, data["history"]["v_loss"], '-', color=colors[name], lw=2, label=f'{name} (val)')
    axes[0].set_title("Validation Loss by Strategy", fontweight='bold')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].spines[['top', 'right']].set_visible(False)

    # Accuracy curves
    for name, data in results.items():
        ep = range(1, len(data["history"]["v_acc"]) + 1)
        axes[1].plot(ep, data["history"]["v_acc"], '-', color=colors[name], lw=2, label=f'{name}')
    axes[1].set_title("Validation Accuracy by Strategy", fontweight='bold')
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].spines[['top', 'right']].set_visible(False)

    # Bar comparison
    names = list(results.keys())
    accs = [results[n]["best_acc"] for n in names]
    bars = axes[2].bar(names, accs, color=[colors[n] for n in names], edgecolor='white', width=0.5)
    for bar, v in zip(bars, accs):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v + 0.005, f'{v:.4f}',
                     ha='center', fontweight='bold', fontsize=11)
    best_idx = np.argmax(accs)
    bars[best_idx].set_edgecolor('#FFD600'); bars[best_idx].set_linewidth(3)
    axes[2].set_title("Best Val Accuracy (star=winner)", fontweight='bold')
    axes[2].set_ylabel("Accuracy"); axes[2].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_strategy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_strategy_comparison.png")
