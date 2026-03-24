"""
Day 11: ICU Mortality — Model Training
Polynomial Regression deep-dive + GPU Neural Net regressor
"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from torch.amp import autocast, GradScaler
import joblib, config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# POLYNOMIAL REGRESSION (sklearn)
# ═══════════════════════════════════════════════════════════════

def train_polynomial_models(X_train, y_train):
    """
    Train Linear, Poly(2), Poly(3) regression with Ridge regularization.
    Polynomial features transform: [x1, x2] → [1, x1, x2, x1², x1·x2, x2²]
    This lets linear regression capture NON-LINEAR relationships.
    """
    logger.info("=" * 60)
    logger.info("POLYNOMIAL REGRESSION — Degree Comparison")
    logger.info("=" * 60)

    cv = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    results = {}

    for degree in config.POLY_DEGREES:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)),
            ("reg", Ridge(alpha=1.0))
        ])

        t = time.time()
        # Negative MSE (sklearn convention: higher=better)
        mse_scores = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
        r2_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)

        pipe.fit(X_train, y_train)
        n_features = pipe.named_steps["poly"].get_feature_names_out().shape[0]
        elapsed = time.time() - t

        results[degree] = {
            "model": pipe,
            "cv_rmse_mean": np.sqrt(mse_scores).mean(),
            "cv_rmse_std": np.sqrt(mse_scores).std(),
            "cv_r2_mean": r2_scores.mean(),
            "cv_r2_std": r2_scores.std(),
            "n_features": n_features,
        }

        logger.info(f"  Degree {degree}: {n_features:>5d} features | "
                     f"CV RMSE={results[degree]['cv_rmse_mean']:.4f}±{results[degree]['cv_rmse_std']:.4f} | "
                     f"R²={results[degree]['cv_r2_mean']:.4f} | {elapsed:.1f}s")

    _plot_degree_comparison(results)
    return results


def _plot_degree_comparison(results):
    """Visualize polynomial degree impact."""
    degrees = list(results.keys())
    rmses = [results[d]["cv_rmse_mean"] for d in degrees]
    r2s = [results[d]["cv_r2_mean"] for d in degrees]
    n_feats = [results[d]["n_features"] for d in degrees]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # RMSE vs degree
    axes[0].bar(degrees, rmses, color=['#4FC3F7', '#66BB6A', '#FF7043'], edgecolor='white', width=0.5)
    axes[0].set_xlabel("Polynomial Degree"); axes[0].set_ylabel("CV RMSE")
    axes[0].set_title("🏥 RMSE vs Polynomial Degree", fontweight='bold')
    axes[0].set_xticks(degrees)
    axes[0].spines[['top','right']].set_visible(False)

    # R² vs degree
    axes[1].bar(degrees, r2s, color=['#4FC3F7', '#66BB6A', '#FF7043'], edgecolor='white', width=0.5)
    axes[1].set_xlabel("Polynomial Degree"); axes[1].set_ylabel("CV R²")
    axes[1].set_title("📈 R² vs Polynomial Degree", fontweight='bold')
    axes[1].set_xticks(degrees)
    axes[1].spines[['top','right']].set_visible(False)

    # Feature explosion
    axes[2].bar(degrees, n_feats, color=['#4FC3F7', '#66BB6A', '#FF7043'], edgecolor='white', width=0.5)
    for i, (d, nf) in enumerate(zip(degrees, n_feats)):
        axes[2].text(d, nf + 50, f'{nf:,}', ha='center', fontweight='bold', fontsize=10)
    axes[2].set_xlabel("Polynomial Degree"); axes[2].set_ylabel("# Features")
    axes[2].set_title("💥 Feature Explosion (Curse of Dimensionality)", fontweight='bold')
    axes[2].set_xticks(degrees)
    axes[2].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_poly_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_poly_comparison.png")


# ═══════════════════════════════════════════════════════════════
# GPU NEURAL NET REGRESSOR (PyTorch)
# ═══════════════════════════════════════════════════════════════

class ICUNet(nn.Module):
    """Simple feedforward regressor for GPU training."""
    def __init__(self, input_dim):
        super().__init__()
        h = config.NN_HIDDEN
        self.net = nn.Sequential(
            nn.Linear(input_dim, h[0]),
            nn.BatchNorm1d(h[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(h[0], h[1]),
            nn.BatchNorm1d(h[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(h[1], 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_gpu_neural_net(X_train, y_train):
    """Train neural net regressor on GPU with AMP."""
    logger.info("=" * 60)
    logger.info(f"GPU NEURAL NET REGRESSOR (device={config.DEVICE})")
    logger.info("=" * 60)

    # Convert to tensors
    X_t = torch.from_numpy(X_train).to(config.DEVICE)
    y_t = torch.from_numpy(y_train).to(config.DEVICE)

    model = ICUNet(X_train.shape[1]).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.NN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    use_amp = config.DEVICE == "cuda"
    scaler = GradScaler(enabled=use_amp)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.NN_BATCH, shuffle=True,
                                          drop_last=True, pin_memory=False)

    # Split for validation (last 15%)
    n_val = int(len(X_t) * 0.15)
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]

    best_loss, patience, best_state = float('inf'), 0, None
    history = {"train_loss": [], "val_loss": []}

    start = time.time()
    for epoch in range(config.NN_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=config.DEVICE, enabled=use_amp):
                pred = model(xb)
                loss = criterion(pred, yb)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        train_loss = epoch_loss / len(dataset)

        # Validate
        model.eval()
        with torch.no_grad():
            with autocast(device_type=config.DEVICE, enabled=use_amp):
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:>3d}/{config.NN_EPOCHS} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        if patience >= 10:
            logger.info(f"  Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(config.DEVICE)

    logger.info(f"  Done in {time.time()-start:.1f}s | Best val MSE: {best_loss:.5f}")
    _plot_nn_history(history)
    return model, history


def _plot_nn_history(history):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], '-', color='#FF7043', lw=2, label='Train MSE')
    ax.plot(epochs, history["val_loss"], '-', color='#4FC3F7', lw=2, label='Val MSE')
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("🧠 Neural Net Training — Loss Curves", fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_nn_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_nn_training.png")


def save_models(poly_results, nn_model, scaler):
    path = f"{config.MODEL_DIR}/day11_models.joblib"
    save_dict = {
        "poly_models": {d: r["model"] for d, r in poly_results.items()},
        "scaler": scaler,
    }
    joblib.dump(save_dict, path, compress=3)
    torch.save(nn_model.state_dict(), f"{config.MODEL_DIR}/day11_nn.pth")
    logger.info(f"  Saved: {path}")
