"""Day 19: Radiosensitivity — Model Training"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import joblib, config

logger = logging.getLogger(__name__)


def train_elastic_net_all_strategies(splits):
    """Elastic Net on all 3 validation strategies -- the core experiment."""
    logger.info("=" * 60)
    logger.info("ELASTIC NET -- 3 Validation Strategies")
    logger.info("=" * 60)

    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    all_results = {}

    for strat, (X_tr, X_te, y_tr, y_te) in splits.items():
        best_score, best_params = -np.inf, {}
        for alpha in config.ELASTIC_ALPHAS:
            for l1 in config.ELASTIC_L1_RATIOS:
                en = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=3000, random_state=config.RANDOM_SEED)
                scores = cross_val_score(en, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=-1)
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_params = {"alpha": alpha, "l1_ratio": l1}

        model = ElasticNet(**best_params, max_iter=3000, random_state=config.RANDOM_SEED)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        all_results[strat] = {
            "model": model, "cv_r2": best_score,
            "test_r2": r2_score(y_te, y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_te, y_pred)),
            "params": best_params, "y_pred": y_pred, "y_true": y_te
        }
        logger.info(f"  {strat:18s} | CV R2={best_score:.4f} | Test R2={all_results[strat]['test_r2']:.4f} "
                     f"| RMSE={all_results[strat]['test_rmse']:.4f} | {best_params}")

    _plot_cross_validation(all_results)
    return all_results


def _plot_cross_validation(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = list(results.keys())
    colors = ['#4FC3F7', '#66BB6A', '#EF5350']

    # R2 comparison
    cv_r2 = [results[n]["cv_r2"] for n in names]
    test_r2 = [results[n]["test_r2"] for n in names]
    x = np.arange(len(names))
    axes[0].bar(x - 0.15, cv_r2, 0.3, color=colors, edgecolor='white', alpha=0.6, label='CV R2')
    axes[0].bar(x + 0.15, test_r2, 0.3, color=colors, edgecolor='white', label='Test R2')
    axes[0].set_xticks(x); axes[0].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    axes[0].set_ylabel("R2"); axes[0].set_title("R2: CV vs Test", fontweight='bold')
    axes[0].legend(); axes[0].spines[['top', 'right']].set_visible(False)

    # RMSE comparison
    rmses = [results[n]["test_rmse"] for n in names]
    axes[1].bar(names, rmses, color=colors, edgecolor='white')
    for i, v in enumerate(rmses):
        axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')
    axes[1].set_ylabel("Test RMSE"); axes[1].set_title("RMSE by Strategy", fontweight='bold')
    axes[1].spines[['top', 'right']].set_visible(False)

    # Cross-dataset scatter
    d = results["cross_A_to_B"]
    axes[2].scatter(d["y_true"], d["y_pred"], alpha=0.3, s=10, color='#EF5350', rasterized=True)
    axes[2].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[2].set_xlabel("Actual SF2"); axes[2].set_ylabel("Predicted SF2")
    axes[2].set_title(f"Cross-Dataset A->B (R2={d['test_r2']:.4f})", fontweight='bold')
    axes[2].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_cross_validation.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_cross_validation.png")


def train_gpu_nn(X_tr, y_tr):
    """GPU neural net on within-A dataset."""
    logger.info("=" * 60)
    logger.info(f"GPU NEURAL NET (device={config.DEVICE})")
    logger.info("=" * 60)

    X_t = torch.from_numpy(X_tr).to(config.DEVICE)
    y_t = torch.from_numpy(y_tr).to(config.DEVICE)

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
        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1:>3d} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")
        if patience >= 10:
            logger.info(f"  Early stop at epoch {epoch+1}"); break

    if best_state:
        model.load_state_dict(best_state); model = model.to(config.DEVICE)
    logger.info(f"  Done in {time.time()-t0:.1f}s | Best val MSE: {best_loss:.5f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["train"], '-', color='#FF7043', lw=2, label='Train')
    ax.plot(history["val"], '-', color='#4FC3F7', lw=2, label='Val')
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title("GPU NN Training", fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_nn_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_nn_training.png")
    return model


def save_models(en_results, nn_model, scaler):
    joblib.dump({"elastic_net": {n: d["model"] for n, d in en_results.items()}, "scaler": scaler},
                f"{config.MODEL_DIR}/day19_sklearn.joblib", compress=3)
    torch.save(nn_model.state_dict(), f"{config.MODEL_DIR}/day19_nn.pth")
    logger.info("  Models saved")
