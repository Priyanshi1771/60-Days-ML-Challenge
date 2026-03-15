"""Day 15: BMI Prediction — Model Training"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
import joblib, config

logger = logging.getLogger(__name__)


def train_random_forest(X_train, y_train):
    """RF with GridSearchCV — trees naturally capture feature interactions."""
    logger.info("=" * 60)
    logger.info("RANDOM FOREST REGRESSOR — GridSearchCV")
    logger.info("=" * 60)

    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    grid = GridSearchCV(
        RandomForestRegressor(random_state=config.RANDOM_SEED, n_jobs=-1),
        config.RF_PARAM_GRID, cv=cv, scoring="neg_mean_squared_error",
        refit=True, n_jobs=-1, return_train_score=True)

    t0 = time.time()
    grid.fit(X_train, y_train)
    logger.info(f"  Done in {time.time()-t0:.1f}s")
    logger.info(f"  Best params: {grid.best_params_}")
    logger.info(f"  Best CV RMSE: {np.sqrt(-grid.best_score_):.4f}")

    # Plot: n_estimators vs performance for best max_depth
    _plot_rf_analysis(grid)

    return grid.best_estimator_, grid


def _plot_rf_analysis(grid):
    results = grid.cv_results_
    best_depth = grid.best_params_["max_depth"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (0) Trees vs RMSE for each max_depth
    for depth in config.RF_PARAM_GRID["max_depth"]:
        mask = np.array(results["param_max_depth"]) == depth
        for leaf in config.RF_PARAM_GRID["min_samples_leaf"][:1]:  # just first leaf setting
            mask2 = mask & (np.array(results["param_min_samples_leaf"]) == leaf)
            if mask2.sum() == 0:
                continue
            n_trees = np.array(results["param_n_estimators"])[mask2].astype(int)
            rmse = np.sqrt(-np.array(results["mean_test_score"])[mask2])
            label = f"depth={depth}" if depth else "depth=None"
            axes[0].plot(n_trees, rmse, 'o-', lw=2, markersize=6, label=label)

    axes[0].set_xlabel("n_estimators"); axes[0].set_ylabel("CV RMSE")
    axes[0].set_title("🌲 RF: Trees vs RMSE by max_depth", fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[0].spines[['top', 'right']].set_visible(False)

    # (1) Train vs CV (overfitting check) for best params
    best_mask = (
        (np.array(results["param_max_depth"]) == best_depth) &
        (np.array(results["param_min_samples_leaf"]) == grid.best_params_["min_samples_leaf"])
    )
    n_trees = np.array(results["param_n_estimators"])[best_mask].astype(int)
    train_rmse = np.sqrt(-np.array(results["mean_train_score"])[best_mask])
    test_rmse = np.sqrt(-np.array(results["mean_test_score"])[best_mask])

    axes[1].plot(n_trees, train_rmse, 's--', color='#FF7043', lw=2, label='Train')
    axes[1].plot(n_trees, test_rmse, 'o-', color='#4FC3F7', lw=2, label='CV Test')
    axes[1].fill_between(n_trees, test_rmse * 0.98, test_rmse * 1.02, alpha=0.1, color='#4FC3F7')
    axes[1].set_xlabel("n_estimators"); axes[1].set_ylabel("RMSE")
    axes[1].set_title("🔍 Overfitting Check (best depth/leaf)", fontweight='bold')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_rf_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_rf_analysis.png")


def train_baselines(X_train, y_train):
    logger.info("=" * 60)
    logger.info("BASELINES")
    logger.info("=" * 60)

    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    trained = {}

    for name, model in [
        ("Ridge (α=1.0)", Ridge(alpha=1.0)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=config.RANDOM_SEED)),
    ]:
        t0 = time.time()
        model.fit(X_train, y_train)
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=cv,
                                         scoring="neg_mean_squared_error", n_jobs=-1))
        trained[name] = {"model": model, "cv_rmse": rmse.mean()}
        logger.info(f"  {name:25s} | RMSE={rmse.mean():.4f} | {time.time()-t0:.1f}s")

    return trained


def train_gpu_nn(X_train, y_train):
    logger.info("=" * 60)
    logger.info(f"GPU NEURAL NET (device={config.DEVICE})")
    logger.info("=" * 60)

    X_t = torch.from_numpy(X_train).to(config.DEVICE)
    y_t = torch.from_numpy(y_train).to(config.DEVICE)

    h = config.NN_HIDDEN
    model = nn.Sequential(
        nn.Linear(X_t.shape[1], h[0]), nn.BatchNorm1d(h[0]), nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(h[0], h[1]), nn.BatchNorm1d(h[1]), nn.ReLU(inplace=True), nn.Dropout(0.2),
        nn.Linear(h[1], 1)
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.NN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    use_amp = config.DEVICE == "cuda"
    scaler = GradScaler(enabled=use_amp)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t),
        batch_size=config.NN_BATCH, shuffle=True, drop_last=True)

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
            logger.info(f"  Epoch {epoch+1:>3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        if patience >= 10:
            logger.info(f"  Early stop at epoch {epoch+1}"); break

    if best_state:
        model.load_state_dict(best_state); model = model.to(config.DEVICE)
    logger.info(f"  Done in {time.time()-t0:.1f}s | Best val MSE: {best_loss:.4f}")

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


def save_models(rf_model, baselines, nn_model, scaler):
    joblib.dump({"rf": rf_model, "baselines": {n: d["model"] for n, d in baselines.items()},
                 "scaler": scaler}, f"{config.MODEL_DIR}/day15_sklearn.joblib", compress=3)
    torch.save(nn_model.state_dict(), f"{config.MODEL_DIR}/day15_nn.pth")
    logger.info("  Models saved")
