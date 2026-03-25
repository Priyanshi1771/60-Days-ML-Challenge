"""Day 21: Pneumonia Detection — Model Training (4-Block CNN on GPU)"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
import config

logger = logging.getLogger(__name__)


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ch = config.CNN_CHANNELS
        self.features = nn.Sequential(
            *self._blk(ch[0], ch[1]), *self._blk(ch[1], ch[2]),
            *self._blk(ch[2], ch[3]), *self._blk(ch[3], ch[4]),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(ch[4] * 16, config.FC_HIDDEN),
            nn.ReLU(inplace=True), nn.Dropout(config.DROPOUT),
            nn.Linear(config.FC_HIDDEN, config.NUM_CLASSES),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _blk(self, ci, co):
        return [nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True), nn.MaxPool2d(2)]

    def forward(self, x):
        return self.classifier(self.features(x))


def train_model(model, train_loader, val_loader):
    logger.info("=" * 60)
    logger.info(f"TRAINING CNN (device={config.DEVICE})")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Params: {n_params:,}")
    logger.info("=" * 60)

    model = model.to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    use_amp = config.DEVICE == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val, patience, best_state = float('inf'), 0, None
    history = {"t_loss": [], "v_loss": [], "t_acc": [], "v_acc": []}

    t0 = time.time()
    for epoch in range(config.EPOCHS):
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
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            r_loss += loss.item() * imgs.size(0)
            total += lbls.size(0); correct += out.argmax(1).eq(lbls).sum().item()

        t_loss, t_acc = r_loss / total, correct / total

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
        else: patience += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:>2d} | Train: loss={t_loss:.4f} acc={t_acc:.3f} | "
                        f"Val: loss={vl:.4f} acc={va:.3f}")
        if patience >= config.PATIENCE:
            logger.info(f"  Early stop at epoch {epoch+1}"); break

    if best_state: model.load_state_dict(best_state); model = model.to(config.DEVICE)
    logger.info(f"  Done in {time.time()-t0:.1f}s")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history["t_loss"]) + 1)
    for ax, t, v, title in [(axes[0], history["t_loss"], history["v_loss"], "Loss"),
                              (axes[1], history["t_acc"], history["v_acc"], "Accuracy")]:
        ax.plot(ep, t, '-', color='#FF7043', lw=2, label='Train')
        ax.plot(ep, v, '-', color='#4FC3F7', lw=2, label='Val')
        ax.set_title(title, fontweight='bold'); ax.legend(); ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_training.png")
    return model, history
