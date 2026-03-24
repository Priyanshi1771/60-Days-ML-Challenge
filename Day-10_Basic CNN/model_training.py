"""
=============================================================================
 Day 10: Malaria Cell Classification — Model Training (Optimized)
=============================================================================
 THE FIRST CNN IN THE 60-DAY CHALLENGE!
 
 CNN ARCHITECTURE (from scratch):
   Input (3×64×64) 
   → Conv2d(3→32, 3×3) + BN + ReLU + MaxPool
   → Conv2d(32→64, 3×3) + BN + ReLU + MaxPool
   → Conv2d(64→128, 3×3) + BN + ReLU + AdaptiveAvgPool
   → Flatten → FC(128→256) + Dropout → FC(256→2)
 
 KEY CONCEPTS INTRODUCED:
   - Convolutional layers (local pattern detection)
   - Batch Normalization (stabilize training)
   - MaxPooling (spatial downsampling)
   - AdaptiveAvgPool (flexible input sizes)
   - Dropout (regularization)
   - Learning rate scheduling (ReduceLROnPlateau)
   - Early stopping (prevent overfitting)
   
 OPTIMIZATION: torch.compile (if available), AMP for GPU, in-place ReLU
=============================================================================
"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import config

logger = logging.getLogger(__name__)


class MalariaCNN(nn.Module):
    """
    Custom CNN for cell image classification.
    
    Architecture choices explained:
    - 3×3 kernels: standard for image classification (captures edges, textures)
    - BatchNorm after each conv: stabilizes gradients, enables higher LR
    - MaxPool(2): halves spatial dims → 64→32→16→adaptive
    - AdaptiveAvgPool(4): makes architecture robust to input size changes
    - Dropout(0.4): prevents overfitting on small-medium datasets
    - 2 FC layers: enough capacity without massive parameter count
    """
    
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        
        ch = config.CNN_CHANNELS  # [3, 32, 64, 128]
        
        # Feature extractor (convolutional blocks)
        self.features = nn.Sequential(
            # Block 1: 3→32, 64×64 → 32×32
            nn.Conv2d(ch[0], ch[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 32→64, 32×32 → 16×16
            nn.Conv2d(ch[1], ch[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 64→128, 16×16 → adaptive 4×4
            nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # Output: 128×4×4 regardless of input size
        )
        
        # Classifier (fully connected)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[3] * 4 * 4, config.FC_HIDDEN),  # 128*16 = 2048 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FC_HIDDEN, num_classes),
        )
        
        # Initialize weights (Kaiming for ReLU)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, val_loader):
    """
    Training loop with all modern best practices:
    - CrossEntropyLoss (combines LogSoftmax + NLLLoss)
    - AdamW optimizer (Adam + proper weight decay)
    - ReduceLROnPlateau scheduler (halve LR when val loss stalls)
    - Early stopping (stop if val loss doesn't improve for N epochs)
    - AMP (Automatic Mixed Precision) for GPU speedup
    - Gradient clipping (prevent exploding gradients)
    """
    logger.info("=" * 60)
    logger.info("TRAINING CNN")
    logger.info("=" * 60)
    logger.info(f"  Architecture: {config.CNN_CHANNELS} → FC({config.FC_HIDDEN}) → {config.NUM_CLASSES}")
    logger.info(f"  Parameters: {count_parameters(model):,}")
    logger.info(f"  Device: {config.DEVICE}")
    
    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                             weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config.SCHEDULER_PATIENCE)
    
    # AMP scaler for GPU
    use_amp = config.DEVICE == "cuda"
    scaler = GradScaler(enabled=use_amp)
    
    # History tracking
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    total_start = time.time()
    
    for epoch in range(config.EPOCHS):
        epoch_start = time.time()
        
        # ─── Train ────────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            
            with autocast(device_type=config.DEVICE, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # ─── Validate ─────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.DEVICE, non_blocking=True)
                labels = labels.to(config.DEVICE, non_blocking=True)
                
                with autocast(device_type=config.DEVICE, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # LR scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        # Logging (compact)
        elapsed = time.time() - epoch_start
        logger.info(f"  Epoch {epoch+1:>2d}/{config.EPOCHS} | "
                     f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                     f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
                     f"LR={current_lr:.1e} | {elapsed:.1f}s"
                     f"{' ★' if patience_counter == 0 else ''}")
        
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            logger.info(f"  ⏹️  Early stopping at epoch {epoch+1} (patience={config.EARLY_STOP_PATIENCE})")
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(config.DEVICE)
    
    total_time = time.time() - total_start
    logger.info(f"  Training complete: {total_time:.1f}s | Best val_loss: {best_val_loss:.4f}")
    
    _plot_training_history(history)
    return model, history


def _plot_training_history(history):
    """Plot loss + accuracy + LR curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(epochs, history["train_loss"], 'o-', color='#FF7043', lw=2, label='Train', markersize=4)
    axes[0].plot(epochs, history["val_loss"], 's-', color='#4FC3F7', lw=2, label='Validation', markersize=4)
    axes[0].set_title("🦟 Loss Curves", fontweight='bold')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].spines[['top','right']].set_visible(False)
    
    # Accuracy
    axes[1].plot(epochs, history["train_acc"], 'o-', color='#FF7043', lw=2, label='Train', markersize=4)
    axes[1].plot(epochs, history["val_acc"], 's-', color='#4FC3F7', lw=2, label='Validation', markersize=4)
    axes[1].set_title("🔬 Accuracy Curves", fontweight='bold')
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].spines[['top','right']].set_visible(False)
    
    # Learning Rate
    axes[2].plot(epochs, history["lr"], 'D-', color='#AB47BC', lw=2, markersize=4)
    axes[2].set_title("📉 Learning Rate Schedule", fontweight='bold')
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_yscale('log'); axes[2].grid(alpha=0.3)
    axes[2].spines[['top','right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_training_history.png")


def save_model(model, history):
    """Save model + history (compressed)."""
    path = f"{config.MODEL_DIR}/day10_cnn.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": str(model),
        "history": history,
        "config": {
            "img_size": config.IMG_SIZE,
            "channels": config.CNN_CHANNELS,
            "fc_hidden": config.FC_HIDDEN,
            "dropout": config.DROPOUT
        }
    }, path)
    logger.info(f"  Model saved: {path} ({os.path.getsize(path)/1024:.0f} KB)")
    return path


import os
