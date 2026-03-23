"""
Day 18: Gene Expression — Model Training
DEEP LEARNING REGRESSION: Compare 5 architectures on high-dimensional genomic data.
This is the first project where DL is the PRIMARY model, not just a comparison baseline.
"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
import config

logger = logging.getLogger(__name__)


def build_model(input_dim, hidden_layers, dropout=config.DROPOUT):
    """Build a feedforward regressor from a list of hidden layer sizes."""
    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_single_model(model, X_train, y_train, X_val, y_val, name="Model"):
    """Train one model on GPU with AMP, early stopping, LR scheduling."""
    model = model.to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    use_amp = config.DEVICE == "cuda"
    scaler = GradScaler(enabled=use_amp)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

    best_val, patience, best_state = float('inf'), 0, None
    history = {"train": [], "val": []}

    for epoch in range(config.EPOCHS):
        # Train
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=config.DEVICE, enabled=use_amp):
                loss = criterion(model(xb).squeeze(-1), yb)
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            ep_loss += loss.item() * xb.size(0)

        train_loss = ep_loss / len(dataset)

        # Validate
        model.eval()
        with torch.no_grad():
            with autocast(device_type=config.DEVICE, enabled=use_amp):
                val_loss = criterion(model(X_val).squeeze(-1), y_val).item()

        scheduler.step(val_loss)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if patience >= config.PATIENCE:
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(config.DEVICE)

    return model, history, best_val


def train_all_architectures(X_train, y_train, X_val, y_val):
    """Train 5 different architectures and compare them."""
    logger.info("=" * 60)
    logger.info(f"DL ARCHITECTURE COMPARISON (device={config.DEVICE})")
    logger.info("=" * 60)

    input_dim = X_train.shape[1]
    results = {}

    for name, hidden in config.ARCHITECTURES.items():
        model = build_model(input_dim, hidden)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        t0 = time.time()
        model, history, best_val = train_single_model(model, X_train, y_train, X_val, y_val, name)
        elapsed = time.time() - t0

        rmse = np.sqrt(best_val)
        results[name] = {
            "model": model, "history": history, "best_val_mse": best_val,
            "rmse": rmse, "n_params": n_params, "time": elapsed,
            "epochs_trained": len(history["train"]), "hidden": hidden
        }
        logger.info(f"  {name:22s} | Params: {n_params:>8,} | Val RMSE: {rmse:.4f} | "
                     f"Epochs: {len(history['train']):>3d} | {elapsed:.1f}s")

    _plot_architecture_comparison(results)
    _plot_training_curves(results)

    return results


def _plot_architecture_comparison(results):
    names = list(results.keys())
    rmses = [results[n]["rmse"] for n in names]
    params = [results[n]["n_params"] for n in names]
    times = [results[n]["time"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # RMSE comparison
    colors = ['#7E57C2', '#4FC3F7', '#66BB6A', '#FF7043', '#EF5350']
    bars = axes[0].bar(range(len(names)), rmses, color=colors, edgecolor='white')
    best_idx = np.argmin(rmses)
    bars[best_idx].set_edgecolor('#00E676'); bars[best_idx].set_linewidth(3)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([n.split('(')[0].strip() for n in names], fontsize=8, rotation=15)
    axes[0].set_ylabel("Val RMSE"); axes[0].set_title("RMSE by Architecture (lower=better)", fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)

    # Params vs RMSE (efficiency curve)
    for i, (n, r, p) in enumerate(zip(names, rmses, params)):
        axes[1].scatter(p, r, s=120, color=colors[i], zorder=5, edgecolors='white', linewidth=1.5)
        axes[1].annotate(n.split('(')[0].strip(), (p, r), fontsize=7, ha='center', va='bottom')
    axes[1].set_xlabel("# Parameters"); axes[1].set_ylabel("Val RMSE")
    axes[1].set_title("Efficiency: Parameters vs Performance", fontweight='bold')
    axes[1].grid(alpha=0.3); axes[1].spines[['top', 'right']].set_visible(False)

    # Training time
    axes[2].barh(range(len(names)), times, color=colors, edgecolor='white')
    axes[2].set_yticks(range(len(names)))
    axes[2].set_yticklabels([n.split('(')[0].strip() for n in names], fontsize=9)
    axes[2].set_xlabel("Training Time (seconds)")
    axes[2].set_title("Training Speed", fontweight='bold')
    axes[2].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_architecture_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_architecture_comparison.png")


def _plot_training_curves(results):
    n_arch = len(results)
    fig, axes = plt.subplots(1, n_arch, figsize=(4 * n_arch, 4))
    if n_arch == 1:
        axes = [axes]
    colors = ['#7E57C2', '#4FC3F7', '#66BB6A', '#FF7043', '#EF5350']

    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        ep = range(1, len(data["history"]["train"]) + 1)
        ax.plot(ep, data["history"]["train"], '-', color='#FF7043', lw=1.5, label='Train', alpha=0.8)
        ax.plot(ep, data["history"]["val"], '-', color='#4FC3F7', lw=1.5, label='Val', alpha=0.8)
        ax.set_title(f"{name.split('(')[0].strip()}\n({data['n_params']:,} params)", fontsize=9, fontweight='bold')
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.legend(fontsize=7); ax.grid(alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle("Training Curves -- All Architectures", fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_training_curves.png")


def train_sklearn_baselines(X_train_np, y_train_np):
    """Quick sklearn baselines for context."""
    logger.info("=" * 60)
    logger.info("SKLEARN BASELINES (for context)")
    logger.info("=" * 60)

    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    baselines = {}
    for name, model in [
        ("Ridge", Ridge(alpha=1.0)),
        ("Lasso", Lasso(alpha=0.01, max_iter=3000)),
        ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=10,
                                                 random_state=config.RANDOM_SEED, n_jobs=-1)),
    ]:
        t0 = time.time()
        model.fit(X_train_np, y_train_np)
        baselines[name] = model
        logger.info(f"  {name:18s} | {time.time()-t0:.1f}s")

    return baselines


def save_models(results, baselines, scaler):
    # Save best DL model
    best_name = min(results, key=lambda k: results[k]["rmse"])
    torch.save({
        "model_state": results[best_name]["model"].state_dict(),
        "architecture": results[best_name]["hidden"],
        "best_rmse": results[best_name]["rmse"]
    }, f"{config.MODEL_DIR}/day18_best_nn.pth")

    import joblib
    joblib.dump({"baselines": baselines, "scaler": scaler},
                f"{config.MODEL_DIR}/day18_sklearn.joblib", compress=3)
    logger.info(f"  Saved best DL model: {best_name}")
