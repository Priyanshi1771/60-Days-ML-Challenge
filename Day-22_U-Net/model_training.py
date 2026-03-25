"""
Day 22: Brain Tumor Segmentation — U-Net Model Training
First SEGMENTATION model in the 60-day challenge!

U-Net Architecture:
  Encoder (contracting): downsample with Conv+Pool, capture context
  Bottleneck: deepest representation
  Decoder (expanding): upsample with ConvTranspose, recover spatial detail
  Skip connections: concatenate encoder features to decoder (preserve details!)

Loss: Dice + BCE combined (handles class imbalance in segmentation)
"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
import config

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Two Conv2d + BN + ReLU (standard U-Net building block)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        enc = config.ENCODER_CH     # [1, 32, 64, 128, 256]
        bneck = config.BOTTLENECK_CH  # 512
        dec = config.DECODER_CH     # [256, 128, 64, 32]

        # Encoder
        self.enc1 = ConvBlock(enc[0], enc[1])   # 1->32
        self.enc2 = ConvBlock(enc[1], enc[2])   # 32->64
        self.enc3 = ConvBlock(enc[2], enc[3])   # 64->128
        self.enc4 = ConvBlock(enc[3], enc[4])   # 128->256
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(enc[4], bneck)  # 256->512

        # Decoder (ConvTranspose2d for upsampling)
        self.up4 = nn.ConvTranspose2d(bneck, dec[0], 2, stride=2)   # 512->256
        self.dec4 = ConvBlock(dec[0] + enc[4], dec[0])               # 256+256->256 (skip!)
        self.up3 = nn.ConvTranspose2d(dec[0], dec[1], 2, stride=2)  # 256->128
        self.dec3 = ConvBlock(dec[1] + enc[3], dec[1])               # 128+128->128
        self.up2 = nn.ConvTranspose2d(dec[1], dec[2], 2, stride=2)  # 128->64
        self.dec2 = ConvBlock(dec[2] + enc[2], dec[2])               # 64+64->64
        self.up1 = nn.ConvTranspose2d(dec[2], dec[3], 2, stride=2)  # 64->32
        self.dec1 = ConvBlock(dec[3] + enc[1], dec[3])               # 32+32->32

        # Output: 1 channel (tumor probability)
        self.out_conv = nn.Conv2d(dec[3], 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder + save skip connections
        e1 = self.enc1(x)                          # (B, 32, 128, 128)
        e2 = self.enc2(self.pool(e1))               # (B, 64, 64, 64)
        e3 = self.enc3(self.pool(e2))               # (B, 128, 32, 32)
        e4 = self.enc4(self.pool(e3))               # (B, 256, 16, 16)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))           # (B, 512, 8, 8)

        # Decoder + skip connections (concatenate!)
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))   # (B, 256, 16, 16)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 128, 32, 32)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 64, 64, 64)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 32, 128, 128)

        return self.out_conv(d1)  # (B, 1, 128, 128) -- raw logits


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss -- standard for medical segmentation.
    Dice handles class imbalance (tiny tumor vs large background).
    BCE provides pixel-level gradient signal."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        flat_p = probs.view(-1)
        flat_t = targets.view(-1)
        intersection = (flat_p * flat_t).sum()
        dice = (2.0 * intersection + self.smooth) / (flat_p.sum() + flat_t.sum() + self.smooth)
        dice_loss = 1.0 - dice

        return bce_loss + dice_loss  # combined


def train_unet(model, train_loader, val_loader):
    logger.info("=" * 60)
    logger.info(f"TRAINING U-NET (device={config.DEVICE})")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,}")
    logger.info("=" * 60)

    model = model.to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
    criterion = DiceBCELoss()
    use_amp = config.DEVICE == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_val, patience, best_state = float('inf'), 0, None
    history = {"t_loss": [], "v_loss": [], "t_dice": [], "v_dice": []}

    t0 = time.time()
    for epoch in range(config.EPOCHS):
        # Train
        model.train()
        r_loss, r_dice, total = 0.0, 0.0, 0
        for imgs, masks in train_loader:
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            masks = masks.to(config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=config.DEVICE, enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, masks)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            r_loss += loss.item() * imgs.size(0)
            # Compute dice for monitoring
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                dice = _dice_score(preds, masks)
                r_dice += dice * imgs.size(0)
            total += imgs.size(0)

        t_loss, t_dice = r_loss / total, r_dice / total

        # Validate
        model.eval()
        v_loss, v_dice, v_total = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(config.DEVICE, non_blocking=True)
                masks = masks.to(config.DEVICE, non_blocking=True)
                with autocast(device_type=config.DEVICE, enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                v_loss += loss.item() * imgs.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                v_dice += _dice_score(preds, masks) * imgs.size(0)
                v_total += imgs.size(0)

        vl, vd = v_loss / v_total, v_dice / v_total
        scheduler.step(vl)

        history["t_loss"].append(t_loss); history["v_loss"].append(vl)
        history["t_dice"].append(t_dice); history["v_dice"].append(vd)

        if vl < best_val:
            best_val = vl; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:>2d} | Train: loss={t_loss:.4f} dice={t_dice:.4f} | "
                        f"Val: loss={vl:.4f} dice={vd:.4f}")
        if patience >= config.PATIENCE:
            logger.info(f"  Early stop at epoch {epoch+1}"); break

    if best_state:
        model.load_state_dict(best_state); model = model.to(config.DEVICE)
    logger.info(f"  Done in {time.time()-t0:.1f}s | Best val dice: {history['v_dice'][np.argmin(history['v_loss'])]:.4f}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history["t_loss"]) + 1)
    axes[0].plot(ep, history["t_loss"], '-', color='#FF7043', lw=2, label='Train')
    axes[0].plot(ep, history["v_loss"], '-', color='#4FC3F7', lw=2, label='Val')
    axes[0].set_title("Dice+BCE Loss", fontweight='bold'); axes[0].legend()
    axes[0].grid(alpha=0.3); axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].plot(ep, history["t_dice"], '-', color='#FF7043', lw=2, label='Train')
    axes[1].plot(ep, history["v_dice"], '-', color='#4FC3F7', lw=2, label='Val')
    axes[1].set_title("Dice Score (higher=better)", fontweight='bold'); axes[1].legend()
    axes[1].grid(alpha=0.3); axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_training.png")

    return model, history


def _dice_score(pred, target, smooth=1.0):
    """Dice coefficient: 2*|P intersection T| / (|P| + |T|). 1.0 = perfect overlap."""
    p = pred.view(-1)
    t = target.view(-1)
    inter = (p * t).sum()
    return ((2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)).item()
