"""Day 17: Readmission Risk — Model Training"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
import joblib, config

logger = logging.getLogger(__name__)


def train_logistic_regression(X_train, y_train):
    """LR with L1/L2 + class_weight='balanced' for imbalanced readmission data."""
    logger.info("=" * 60)
    logger.info("LOGISTIC REGRESSION — GridSearchCV")
    logger.info("=" * 60)

    cv = StratifiedKFold(n_splits=config.N_SPLITS if hasattr(config, 'N_SPLITS') else 5,
                          shuffle=True, random_state=config.RANDOM_SEED)

    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED),
        config.LR_PARAM_GRID, cv=cv, scoring="roc_auc",
        refit=True, n_jobs=-1, return_train_score=True)

    t0 = time.time()
    grid.fit(X_train, y_train)
    logger.info(f"  Done in {time.time()-t0:.1f}s")
    logger.info(f"  Best AUC: {grid.best_score_:.4f}")
    logger.info(f"  Best params: {grid.best_params_}")

    _plot_lr_coefficients(grid.best_estimator_)
    return grid.best_estimator_, grid


def _plot_lr_coefficients(model):
    coefs = model.coef_.ravel()
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    top_n = min(15, len(sorted_idx))
    idx = sorted_idx[:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#EF5350' if coefs[i] > 0 else '#4FC3F7' for i in idx]
    ax.barh(range(top_n), coefs[idx], color=colors, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([config.FEATURE_NAMES[i] for i in idx], fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', lw=1, linestyle='--')
    ax.set_xlabel("Coefficient (log-odds)")
    ax.set_title("🏥 LR Coefficients — Readmission Risk Factors\n(🔴 Red = increases risk | 🔵 Blue = decreases risk)", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_lr_coefficients.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_lr_coefficients.png")


def train_baselines(X_train, y_train):
    logger.info("=" * 60)
    logger.info("BASELINES")
    logger.info("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    trained = {}
    for name, model in [
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=10,
                                                  class_weight='balanced', random_state=config.RANDOM_SEED, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                          random_state=config.RANDOM_SEED)),
    ]:
        t0 = time.time()
        model.fit(X_train, y_train)
        auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        trained[name] = {"model": model, "cv_auc": auc.mean()}
        logger.info(f"  {name:22s} | AUC={auc.mean():.4f} | {time.time()-t0:.1f}s")
    return trained


def compare_temporal_vs_random(df, prepare_fn):
    """
    THE KEY EXPERIMENT: Train model on temporal split vs random split.
    Show that random split gives INFLATED performance (data leakage from future).
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Temporal vs Random Split")
    logger.info("=" * 60)

    from data_pipeline import time_based_split, random_split_for_comparison

    results = {}
    for split_name, split_fn in [("Temporal (correct)", time_based_split),
                                   ("Random (leaky!)", random_split_for_comparison)]:
        train_df, test_df = split_fn(df)
        X_tr, X_te, y_tr, y_te, _ = prepare_fn(train_df, test_df)

        model = LogisticRegression(C=1.0, penalty='l2', solver='saga', class_weight='balanced',
                                    max_iter=2000, random_state=config.RANDOM_SEED)
        model.fit(X_tr, y_tr)

        from sklearn.metrics import roc_auc_score, f1_score
        y_proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_proba)
        f1 = f1_score(y_te, model.predict(X_te))

        results[split_name] = {"auc": auc, "f1": f1, "test_rate": y_te.mean()}
        logger.info(f"  {split_name:25s} | AUC={auc:.4f} | F1={f1:.4f} | Test readmit rate={y_te.mean():.4f}")

    # Show the difference
    auc_diff = results["Random (leaky!)"]["auc"] - results["Temporal (correct)"]["auc"]
    logger.info(f"\n  ⚠️  Random split AUC is {auc_diff:+.4f} {'HIGHER' if auc_diff > 0 else 'lower'} — this is DATA LEAKAGE!")

    _plot_split_comparison(results)
    return results


def _plot_split_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results.keys())
    aucs = [results[n]["auc"] for n in names]
    f1s = [results[n]["f1"] for n in names]
    colors = ['#66BB6A', '#EF5350']

    for ax, metric, vals, ylabel in [(axes[0], "AUC-ROC", aucs, "AUC"),
                                       (axes[1], "F1 Score", f1s, "F1")]:
        bars = ax.bar(names, vals, color=colors, edgecolor='white', width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_title(f"🔀 {metric}: Temporal vs Random Split", fontweight='bold')
        ax.spines[['top','right']].set_visible(False)

    fig.suptitle("⚠️ Random Split INFLATES Performance (Data Leakage from Future!)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_temporal_vs_random.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_temporal_vs_random.png")


def train_gpu_nn(X_train, y_train):
    logger.info("=" * 60)
    logger.info(f"GPU NEURAL NET (device={config.DEVICE})")
    logger.info("=" * 60)

    X_t = torch.from_numpy(X_train).to(config.DEVICE)
    y_t = torch.from_numpy(y_train.astype(np.float32)).to(config.DEVICE)

    # Handle class imbalance with pos_weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(config.DEVICE)

    h = config.NN_HIDDEN
    model = nn.Sequential(
        nn.Linear(X_t.shape[1], h[0]), nn.BatchNorm1d(h[0]), nn.ReLU(inplace=True), nn.Dropout(0.4),
        nn.Linear(h[0], h[1]), nn.BatchNorm1d(h[1]), nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(h[1], 1)
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.NN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
            logger.info(f"  Epoch {epoch+1:>3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        if patience >= 8:
            logger.info(f"  Early stop at epoch {epoch+1}"); break

    if best_state:
        model.load_state_dict(best_state); model = model.to(config.DEVICE)
    logger.info(f"  Done in {time.time()-t0:.1f}s")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["train"], '-', color='#FF7043', lw=2, label='Train')
    ax.plot(history["val"], '-', color='#4FC3F7', lw=2, label='Val')
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title("🧠 NN Training", fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_nn_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_nn_training.png")
    return model


def save_models(lr_model, baselines, nn_model, scaler):
    joblib.dump({"lr": lr_model, "baselines": {n: d["model"] for n, d in baselines.items()},
                 "scaler": scaler}, f"{config.MODEL_DIR}/day17_sklearn.joblib", compress=3)
    torch.save(nn_model.state_dict(), f"{config.MODEL_DIR}/day17_nn.pth")
    logger.info("  Models saved")
