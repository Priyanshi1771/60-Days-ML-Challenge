"""
Day 12: Blood Pressure — Model Training
OLS vs Ridge vs Lasso (multicollinearity showdown) + GPU NN
"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold
from torch.amp import autocast, GradScaler
import joblib, config

logger = logging.getLogger(__name__)


def train_regression_showdown(X_train, y_train, target_name="Systolic"):
    """OLS vs Ridge vs Lasso vs ElasticNet — multicollinearity showdown."""
    logger.info("=" * 60)
    logger.info(f"REGRESSION SHOWDOWN — {target_name} BP")
    logger.info("=" * 60)

    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    results = {}

    # OLS — will suffer from multicollinearity
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    ols_rmse = np.sqrt(-cross_val_score(ols, X_train, y_train, cv=cv,
                                         scoring='neg_mean_squared_error', n_jobs=-1))
    results["OLS (No Reg)"] = {"model": ols, "cv_rmse": ols_rmse.mean(), "cv_std": ols_rmse.std()}
    logger.info(f"  OLS:        RMSE={ols_rmse.mean():.4f} | Coef range=[{ols.coef_.min():.2f}, {ols.coef_.max():.2f}]")

    # Ridge (L2) — stabilizes coefficients
    ridge = RidgeCV(alphas=config.RIDGE_ALPHAS, cv=cv, scoring='neg_mean_squared_error')
    ridge.fit(X_train, y_train)
    ridge_rmse = np.sqrt(-cross_val_score(ridge, X_train, y_train, cv=cv,
                                           scoring='neg_mean_squared_error', n_jobs=-1))
    results["Ridge (L2)"] = {"model": ridge, "cv_rmse": ridge_rmse.mean(), "cv_std": ridge_rmse.std()}
    logger.info(f"  Ridge:      RMSE={ridge_rmse.mean():.4f} | α={ridge.alpha_:.4f}")

    # Lasso (L1) — zeros out redundant features
    lasso = LassoCV(alphas=config.RIDGE_ALPHAS, cv=cv, n_jobs=-1, random_state=config.RANDOM_SEED)
    lasso.fit(X_train, y_train)
    lasso_rmse = np.sqrt(-cross_val_score(lasso, X_train, y_train, cv=cv,
                                           scoring='neg_mean_squared_error', n_jobs=-1))
    n_zeroed = np.sum(np.abs(lasso.coef_) < 1e-6)
    results["Lasso (L1)"] = {"model": lasso, "cv_rmse": lasso_rmse.mean(), "cv_std": lasso_rmse.std()}
    logger.info(f"  Lasso:      RMSE={lasso_rmse.mean():.4f} | Zeroed: {n_zeroed}/{len(lasso.coef_)}")

    # ElasticNet (L1+L2)
    enet = ElasticNetCV(alphas=config.RIDGE_ALPHAS, cv=cv, n_jobs=-1, random_state=config.RANDOM_SEED)
    enet.fit(X_train, y_train)
    enet_rmse = np.sqrt(-cross_val_score(enet, X_train, y_train, cv=cv,
                                          scoring='neg_mean_squared_error', n_jobs=-1))
    results["ElasticNet"] = {"model": enet, "cv_rmse": enet_rmse.mean(), "cv_std": enet_rmse.std()}
    logger.info(f"  ElasticNet: RMSE={enet_rmse.mean():.4f} | α={enet.alpha_:.4f}")

    _plot_coefficient_comparison(results, target_name)
    return results


def _plot_coefficient_comparison(results, target_name):
    """How multicollinearity makes OLS coefficients explode vs Ridge/Lasso."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    names = config.FEATURE_NAMES

    for ax, (model_name, data) in zip(axes.ravel(), results.items()):
        coefs = data["model"].coef_
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        colors = ['#EF5350' if c > 0 else '#42A5F5' for c in coefs[sorted_idx]]
        ax.barh(range(len(names)), coefs[sorted_idx], color=colors, alpha=0.8, edgecolor='white')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([names[i] for i in sorted_idx], fontsize=8)
        ax.axvline(0, color='gray', lw=1, ls='--')
        ax.set_title(f"{model_name} (RMSE={data['cv_rmse']:.3f})", fontweight='bold', fontsize=11)
        ax.invert_yaxis(); ax.spines[['top','right']].set_visible(False)

    fig.suptitle(f"🔬 Coefficient Comparison — {target_name} BP\n"
                 "(OLS explodes from collinearity | Ridge/Lasso stabilize)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_coef_{target_name.lower()}.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: 04_coef_{target_name.lower()}.png")


class BPNet(nn.Module):
    """Multi-output: predicts systolic + diastolic simultaneously."""
    def __init__(self, input_dim):
        super().__init__()
        h = config.NN_HIDDEN
        self.net = nn.Sequential(
            nn.Linear(input_dim, h[0]), nn.BatchNorm1d(h[0]), nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(h[0], h[1]), nn.BatchNorm1d(h[1]), nn.ReLU(inplace=True), nn.Dropout(0.15),
            nn.Linear(h[1], h[2]), nn.ReLU(inplace=True),
            nn.Linear(h[2], 2),
        )
    def forward(self, x):
        return self.net(x)


def train_gpu_nn(X_train, ys_train, yd_train):
    """Train dual-output BP neural net on GPU with AMP."""
    logger.info("=" * 60)
    logger.info(f"GPU NEURAL NET — Multi-Output (device={config.DEVICE})")
    logger.info("=" * 60)

    y_both = np.column_stack([ys_train, yd_train]).astype(np.float32)
    X_t = torch.from_numpy(X_train).to(config.DEVICE)
    y_t = torch.from_numpy(y_both).to(config.DEVICE)

    n_val = int(len(X_t) * 0.15)
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    X_tr, y_tr = X_t[:-n_val], y_t[:-n_val]

    model = BPNet(X_train.shape[1]).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.NN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    use_amp = config.DEVICE == "cuda"
    scaler = GradScaler(enabled=use_amp)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=config.NN_BATCH, shuffle=True, drop_last=True)

    best_loss, patience_cnt, best_state = float('inf'), 0, None
    history = {"train": [], "val": []}

    start = time.time()
    for epoch in range(config.NN_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=config.DEVICE, enabled=use_amp):
                loss = criterion(model(xb), yb)
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        train_loss = epoch_loss / len(X_tr)
        model.eval()
        with torch.no_grad():
            with autocast(device_type=config.DEVICE, enabled=use_amp):
                val_loss = criterion(model(X_val), y_val).item()

        scheduler.step(val_loss)
        history["train"].append(train_loss); history["val"].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss; patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        if (epoch+1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1:>3d} | Train={train_loss:.4f} | Val={val_loss:.4f}")
        if patience_cnt >= 10:
            logger.info(f"  Early stop at epoch {epoch+1}"); break

    if best_state: model.load_state_dict(best_state); model.to(config.DEVICE)
    logger.info(f"  Done in {time.time()-start:.1f}s | Best val MSE: {best_loss:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["train"], color='#FF7043', lw=2, label='Train')
    ax.plot(history["val"], color='#42A5F5', lw=2, label='Val')
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.legend()
    ax.set_title("🧠 GPU Neural Net — Dual Output (Sys+Dia)", fontweight='bold')
    ax.grid(alpha=0.3); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_nn_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_nn_training.png")
    return model


def save_models(sys_results, dia_results, nn_model, scaler):
    joblib.dump({
        "sys": {k: v["model"] for k, v in sys_results.items()},
        "dia": {k: v["model"] for k, v in dia_results.items()},
        "scaler": scaler,
    }, f"{config.MODEL_DIR}/day12_sklearn.joblib", compress=3)
    torch.save(nn_model.state_dict(), f"{config.MODEL_DIR}/day12_nn.pth")
    logger.info("  Models saved")
