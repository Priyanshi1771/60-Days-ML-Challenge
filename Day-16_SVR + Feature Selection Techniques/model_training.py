"""Day 16: Telomere Length — Model Training: SVR + Feature Selection + GPU NN"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
import joblib, config

logger = logging.getLogger(__name__)


def train_svr_per_selection(X_train, y_train, selection_results):
    """
    Train SVR on features selected by EACH method → proves which selection is best.
    SVR + RBF kernel is sensitive to irrelevant features → feature selection matters!
    """
    logger.info("=" * 60)
    logger.info("SVR — Trained Per Feature Selection Method")
    logger.info("=" * 60)

    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    svr_results = {}

    for method, sel_data in selection_results.items():
        mask = sel_data["mask"]
        X_sel = X_train[:, mask]

        # Quick SVR with default params for comparison
        svr = SVR(kernel="rbf", C=10.0, epsilon=0.05)
        rmse = np.sqrt(-cross_val_score(svr, X_sel, y_train, cv=cv,
                                         scoring="neg_mean_squared_error", n_jobs=-1))
        r2 = cross_val_score(svr, X_sel, y_train, cv=cv, scoring="r2", n_jobs=-1)

        svr_results[method] = {
            "cv_rmse": rmse.mean(), "cv_rmse_std": rmse.std(),
            "cv_r2": r2.mean(), "n_features": int(mask.sum())
        }
        logger.info(f"  {method:15s} ({mask.sum():>2d} feats) | RMSE={rmse.mean():.4f} | R²={r2.mean():.4f}")

    _plot_selection_vs_performance(svr_results)
    return svr_results


def _plot_selection_vs_performance(svr_results):
    methods = list(svr_results.keys())
    rmses = [svr_results[m]["cv_rmse"] for m in methods]
    r2s = [svr_results[m]["cv_r2"] for m in methods]
    n_feats = [svr_results[m]["n_features"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#EF5350', '#FF7043', '#FFB74D', '#66BB6A', '#4FC3F7', '#7E57C2'][:len(methods)]

    # RMSE per method
    bars = axes[0].bar(methods, rmses, color=colors, edgecolor='white')
    for bar, r, nf in zip(bars, rmses, n_feats):
        axes[0].text(bar.get_x() + bar.get_width()/2, r + 0.002, f'{r:.4f}\n({nf}f)',
                     ha='center', fontsize=8, fontweight='bold')
    axes[0].set_ylabel("CV RMSE"); axes[0].set_title("🧬 SVR RMSE by Selection Method", fontweight='bold')
    axes[0].tick_params(axis='x', rotation=25); axes[0].spines[['top', 'right']].set_visible(False)

    # R² per method
    bars = axes[1].bar(methods, r2s, color=colors, edgecolor='white')
    best_idx = np.argmax(r2s)
    bars[best_idx].set_edgecolor('#00E676'); bars[best_idx].set_linewidth(3)
    for bar, r in zip(bars, r2s):
        axes[1].text(bar.get_x() + bar.get_width()/2, max(0, r) + 0.005, f'{r:.4f}',
                     ha='center', fontsize=9, fontweight='bold')
    axes[1].set_ylabel("CV R²"); axes[1].set_title("📈 SVR R² by Selection Method (★=best)", fontweight='bold')
    axes[1].tick_params(axis='x', rotation=25); axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_selection_performance.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_selection_performance.png")


def train_best_svr(X_train, y_train, best_mask):
    """Full GridSearchCV on best feature subset."""
    logger.info("=" * 60)
    logger.info("SVR — Full GridSearch on Best Feature Subset")
    logger.info("=" * 60)

    X_sel = X_train[:, best_mask]
    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    grid = GridSearchCV(SVR(), config.SVR_PARAM_GRID, cv=cv,
                         scoring="neg_mean_squared_error", refit=True, n_jobs=-1)

    t0 = time.time()
    grid.fit(X_sel, y_train)
    logger.info(f"  Done in {time.time()-t0:.1f}s | Best RMSE: {np.sqrt(-grid.best_score_):.4f}")
    logger.info(f"  Best params: {grid.best_params_}")

    return grid.best_estimator_, grid


def train_baselines(X_train, y_train, best_mask):
    """Baselines on best feature subset."""
    logger.info("=" * 60)
    logger.info("BASELINES")
    logger.info("=" * 60)

    X_sel = X_train[:, best_mask]
    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    trained = {}

    for name, model in [
        ("Ridge", Ridge(alpha=1.0)),
        ("Random Forest", RandomForestRegressor(n_estimators=200, max_depth=10,
                                                 random_state=config.RANDOM_SEED, n_jobs=-1)),
    ]:
        model.fit(X_sel, y_train)
        rmse = np.sqrt(-cross_val_score(model, X_sel, y_train, cv=cv,
                                         scoring="neg_mean_squared_error", n_jobs=-1))
        trained[name] = {"model": model, "cv_rmse": rmse.mean()}
        logger.info(f"  {name:20s} | RMSE={rmse.mean():.4f}")

    return trained


def train_gpu_nn(X_train, y_train, best_mask):
    """GPU NN on best feature subset."""
    logger.info("=" * 60)
    logger.info(f"GPU NEURAL NET (device={config.DEVICE})")
    logger.info("=" * 60)

    X_sel = X_train[:, best_mask]
    X_t = torch.from_numpy(X_sel).to(config.DEVICE)
    y_t = torch.from_numpy(y_train).to(config.DEVICE)

    h = config.NN_HIDDEN
    model = nn.Sequential(
        nn.Linear(X_sel.shape[1], h[0]), nn.BatchNorm1d(h[0]), nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(h[0], h[1]), nn.BatchNorm1d(h[1]), nn.ReLU(inplace=True), nn.Dropout(0.2),
        nn.Linear(h[1], 1)
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.NN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    use_amp = config.DEVICE == "cuda"
    scaler = GradScaler(enabled=use_amp)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t), batch_size=config.NN_BATCH, shuffle=True, drop_last=True)

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

        train_loss = ep_loss / len(X_t)
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
            logger.info(f"  Epoch {epoch+1:>3d} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")
        if patience >= 10:
            logger.info(f"  Early stop at epoch {epoch+1}"); break

    if best_state:
        model.load_state_dict(best_state); model = model.to(config.DEVICE)
    logger.info(f"  Done in {time.time()-t0:.1f}s")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["train"], '-', color='#FF7043', lw=2, label='Train')
    ax.plot(history["val"], '-', color='#4FC3F7', lw=2, label='Val')
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title("🧠 NN Training", fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_nn_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_nn_training.png")
    return model


def save_models(svr_model, baselines, nn_model, scaler, best_mask):
    joblib.dump({"svr": svr_model, "baselines": {n: d["model"] for n, d in baselines.items()},
                 "scaler": scaler, "feature_mask": best_mask},
                f"{config.MODEL_DIR}/day16_sklearn.joblib", compress=3)
    torch.save(nn_model.state_dict(), f"{config.MODEL_DIR}/day16_nn.pth")
    logger.info("  Models saved")
