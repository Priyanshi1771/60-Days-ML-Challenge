"""Day 20: Viral Load — Model Training: LSTM vs GRU on GPU"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
import config

logger = logging.getLogger(__name__)


class SeqModel(nn.Module):
    def __init__(self, rnn_type="LSTM", input_size=1, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        RNN = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = RNN(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _train_one(X_train, y_train, rnn_type):
    X_t = torch.from_numpy(X_train).to(config.DEVICE)
    y_t = torch.from_numpy(y_train).to(config.DEVICE)

    hidden = config.LSTM_HIDDEN if rnn_type == "LSTM" else config.GRU_HIDDEN
    layers = config.LSTM_LAYERS if rnn_type == "LSTM" else config.GRU_LAYERS

    model = SeqModel(rnn_type, hidden=hidden, layers=layers, dropout=config.DROPOUT).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    use_amp = config.DEVICE == "cuda"
    scaler = GradScaler(enabled=use_amp)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t),
        batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

    n_val = max(1, int(len(X_t) * 0.15))
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    best_loss, patience, best_state = float('inf'), 0, None
    history = {"train": [], "val": []}

    t0 = time.time()
    for epoch in range(config.EPOCHS):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=config.DEVICE, enabled=use_amp):
                loss = criterion(model(xb), yb)
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            ep_loss += loss.item() * xb.size(0)

        train_loss = ep_loss / len(X_t)
        model.eval()
        with torch.no_grad():
            with autocast(device_type=config.DEVICE, enabled=use_amp):
                val_loss = criterion(model(X_val), y_val).item()
        scheduler.step(val_loss)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 10:
            break

    if best_state:
        model.load_state_dict(best_state); model = model.to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  {rnn_type}: {n_params:,} params | Val RMSE: {np.sqrt(best_loss):.5f} | "
                f"Epochs: {len(history['train'])} | {time.time()-t0:.1f}s")
    return model, history, best_loss


def train_all_models(X_train, y_train):
    logger.info("=" * 60)
    logger.info(f"SEQUENTIAL MODELS (device={config.DEVICE})")
    logger.info("=" * 60)

    results = {}
    for rnn_type in ["LSTM", "GRU"]:
        model, history, best_val = _train_one(X_train, y_train, rnn_type)
        results[rnn_type] = {"model": model, "history": history, "best_val": best_val}

    # Plot LSTM vs GRU
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, color in [("LSTM", "#7E57C2"), ("GRU", "#FF7043")]:
        h = results[name]["history"]
        axes[0].plot(h["train"], '-', color=color, lw=1.5, label=f'{name} Train', alpha=0.7)
        axes[0].plot(h["val"], '--', color=color, lw=2, label=f'{name} Val')
    axes[0].set_title("Training Curves: LSTM vs GRU", fontweight='bold')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].spines[['top', 'right']].set_visible(False)

    names = list(results.keys())
    vals = [np.sqrt(results[n]["best_val"]) for n in names]
    axes[1].bar(names, vals, color=['#7E57C2', '#FF7043'], edgecolor='white', width=0.4)
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.001, f'{v:.5f}', ha='center', fontweight='bold')
    axes[1].set_ylabel("Val RMSE"); axes[1].set_title("Best Validation RMSE", fontweight='bold')
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_lstm_vs_gru.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_lstm_vs_gru.png")
    return results


def save_models(results):
    for name, data in results.items():
        torch.save(data["model"].state_dict(), f"{config.MODEL_DIR}/day20_{name.lower()}.pth")
    logger.info("  Models saved")
