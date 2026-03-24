"""
Day 14: Drug Response — Model Training
Lasso deep-dive (L1 feature selection on TF-IDF) + GPU Neural Net
"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, cross_val_score
import joblib, config

logger = logging.getLogger(__name__)


def train_lasso_sweep(X_train, y_train):
    """
    Lasso (L1) key property: drives irrelevant feature weights to EXACTLY ZERO.
    With 5000 TF-IDF features, most words are irrelevant → Lasso auto-selects.
    """
    logger.info("=" * 60)
    logger.info("LASSO ALPHA SWEEP — L1 Feature Selection")
    logger.info("=" * 60)

    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    results = []

    for alpha in config.LASSO_ALPHAS:
        model = Lasso(alpha=alpha, max_iter=3000, random_state=config.RANDOM_SEED)
        t0 = time.time()
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=cv,
                                         scoring="neg_mean_squared_error", n_jobs=-1))
        model.fit(X_train, y_train)

        n_nonzero = np.sum(model.coef_ != 0)
        n_total = len(model.coef_)

        results.append({
            "alpha": alpha, "cv_rmse": rmse.mean(), "cv_rmse_std": rmse.std(),
            "nonzero": n_nonzero, "total": n_total,
            "sparsity": 1 - n_nonzero / n_total, "model": model
        })
        logger.info(f"  α={alpha:<7.4f} | RMSE={rmse.mean():.4f} | "
                     f"Features: {n_nonzero:>4d}/{n_total} ({n_nonzero/n_total*100:>5.1f}%) | {time.time()-t0:.1f}s")

    _plot_lasso_sweep(results)
    return results


def _plot_lasso_sweep(results):
    alphas = [r["alpha"] for r in results]
    rmses = [r["cv_rmse"] for r in results]
    nonzeros = [r["nonzero"] for r in results]
    sparsities = [r["sparsity"] * 100 for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # RMSE vs alpha
    axes[0].semilogx(alphas, rmses, 'o-', color='#EF5350', lw=2.5, markersize=8)
    best_idx = np.argmin(rmses)
    axes[0].scatter([alphas[best_idx]], [rmses[best_idx]], s=200, color='#66BB6A', zorder=5, marker='*')
    axes[0].set_xlabel("Lasso α (log)"); axes[0].set_ylabel("CV RMSE")
    axes[0].set_title(f"📈 RMSE vs α — Best at α={alphas[best_idx]}", fontweight='bold')
    axes[0].grid(alpha=0.3); axes[0].spines[['top','right']].set_visible(False)

    # Nonzero features vs alpha
    axes[1].semilogx(alphas, nonzeros, 's-', color='#4FC3F7', lw=2.5, markersize=8)
    axes[1].set_xlabel("Lasso α (log)"); axes[1].set_ylabel("# Non-zero Features")
    axes[1].set_title("🔪 Feature Selection — Lasso Zeroes Out Words", fontweight='bold')
    axes[1].grid(alpha=0.3); axes[1].spines[['top','right']].set_visible(False)

    # Sparsity % vs alpha
    axes[2].semilogx(alphas, sparsities, 'D-', color='#AB47BC', lw=2.5, markersize=8)
    axes[2].set_xlabel("Lasso α (log)"); axes[2].set_ylabel("Sparsity (%)")
    axes[2].set_title("💨 Sparsity — % of Features Killed", fontweight='bold')
    axes[2].grid(alpha=0.3); axes[2].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_lasso_sweep.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_lasso_sweep.png")


def train_baselines(X_train, y_train):
    logger.info("=" * 60)
    logger.info("BASELINE MODELS")
    logger.info("=" * 60)

    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    trained = {}

    for name, model in [
        ("Ridge (α=1.0)", Ridge(alpha=1.0)),
        ("ElasticNet (α=0.01)", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=3000)),
    ]:
        t0 = time.time()
        model.fit(X_train, y_train)
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=cv,
                                         scoring="neg_mean_squared_error", n_jobs=-1))
        trained[name] = {"model": model, "cv_rmse": rmse.mean()}
        logger.info(f"  {name:25s} | RMSE={rmse.mean():.4f} | {time.time()-t0:.1f}s")

    return trained


def train_gpu_nn(X_train_sparse, y_train):
    """GPU neural net on dense version of TF-IDF + numeric features."""
    logger.info("=" * 60)
    logger.info(f"GPU NEURAL NET (device={config.DEVICE})")
    logger.info("=" * 60)

    # Convert sparse → dense (for neural net only)
    X_dense = X_train_sparse.toarray().astype(np.float32)
    X_t = torch.from_numpy(X_dense).to(config.DEVICE)
    y_t = torch.from_numpy(y_train).to(config.DEVICE)
    del X_dense  # free memory

    h = config.NN_HIDDEN
    model = nn.Sequential(
        nn.Linear(X_t.shape[1], h[0]), nn.BatchNorm1d(h[0]), nn.ReLU(inplace=True), nn.Dropout(0.4),
        nn.Linear(h[0], h[1]), nn.BatchNorm1d(h[1]), nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(h[1], 1)
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.NN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    use_amp = config.DEVICE == "cuda"
    scaler = GradScaler(enabled=use_amp)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.NN_BATCH, shuffle=True, drop_last=True)

    n_val = max(1, int(len(X_t) * 0.15))
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]

    best_loss, patience, best_state = float('inf'), 0, None
    history = {"train": [], "val": []}

    t0 = time.time()
    for epoch in range(config.NN_EPOCHS):
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
        model.eval()
        with torch.no_grad():
            with autocast(device_type=config.DEVICE, enabled=use_amp):
                val_loss = criterion(model(X_val).squeeze(-1), y_val).item()

        scheduler.step(val_loss)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1:>3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        if patience >= 8:
            logger.info(f"  Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state); model = model.to(config.DEVICE)
    logger.info(f"  Done in {time.time()-t0:.1f}s | Best val MSE: {best_loss:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["train"], '-', color='#FF7043', lw=2, label='Train')
    ax.plot(history["val"], '-', color='#4FC3F7', lw=2, label='Val')
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title("🧠 NN Training", fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_nn_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_nn_training.png")
    return model


def save_models(lasso_results, baselines, nn_model, tfidf, scaler):
    best_lasso = min(lasso_results, key=lambda r: r["cv_rmse"])
    joblib.dump({
        "best_lasso": best_lasso["model"],
        "baselines": {n: d["model"] for n, d in baselines.items()},
        "tfidf": tfidf, "scaler": scaler
    }, f"{config.MODEL_DIR}/day14_sklearn.joblib", compress=3)
    torch.save(nn_model.state_dict(), f"{config.MODEL_DIR}/day14_nn.pth")
    logger.info("  Models saved")
