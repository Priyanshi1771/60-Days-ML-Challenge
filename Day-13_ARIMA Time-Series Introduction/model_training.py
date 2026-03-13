"""
Day 13: COVID-19 Forecasting — Model Training
ARIMA (classical) + Exponential Smoothing + GPU LSTM (deep learning)
"""
import logging, time, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
import joblib, config

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# ARIMA — Auto-Regressive Integrated Moving Average
# ═══════════════════════════════════════════════════════════════

def train_arima(train_df, test_df):
    """
    ARIMA(p,d,q):
      AR(p): current value depends on p past values
      I(d):  differencing d times to make series stationary
      MA(q): current value depends on q past forecast errors

    We grid-search (p,d,q) minimizing AIC on training data.
    """
    logger.info("=" * 60)
    logger.info("ARIMA — Grid Search for Best (p,d,q)")
    logger.info("=" * 60)

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        logger.warning("statsmodels not installed — skipping ARIMA")
        return None, None

    train_vals = train_df["cases"].values.astype(np.float64)
    best_aic, best_order, best_model = np.inf, (1, 1, 1), None

    t0 = time.time()
    for p in config.ARIMA_P_RANGE:
        for d in config.ARIMA_D_RANGE:
            for q in config.ARIMA_Q_RANGE:
                try:
                    model = ARIMA(train_vals, order=(p, d, q))
                    fit = model.fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                        best_model = fit
                except:
                    continue

    logger.info(f"  Best order: ARIMA{best_order} | AIC={best_aic:.1f} | {time.time()-t0:.1f}s")

    # Forecast
    n_test = len(test_df)
    forecast = best_model.forecast(steps=n_test)
    forecast = np.clip(forecast, 100, None).astype(np.float32)

    logger.info(f"  Forecasted {n_test} days ahead")
    return best_model, forecast


def train_exponential_smoothing(train_df, test_df):
    """Holt-Winters Exponential Smoothing with weekly seasonality."""
    logger.info("=" * 60)
    logger.info("EXPONENTIAL SMOOTHING (Holt-Winters)")
    logger.info("=" * 60)

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        logger.warning("statsmodels not installed — skipping ETS")
        return None, None

    train_vals = train_df["cases"].values.astype(np.float64)
    t0 = time.time()

    model = ExponentialSmoothing(
        train_vals, trend='add', seasonal='add',
        seasonal_periods=config.ETS_SEASONAL_PERIOD,
        initialization_method='estimated'
    ).fit(optimized=True)

    n_test = len(test_df)
    forecast = model.forecast(n_test)
    forecast = np.clip(forecast, 100, None).astype(np.float32)

    logger.info(f"  AIC={model.aic:.1f} | Seasonal period={config.ETS_SEASONAL_PERIOD} | {time.time()-t0:.1f}s")
    return model, forecast


# ═══════════════════════════════════════════════════════════════
# GPU LSTM — Sequence-to-One Forecaster
# ═══════════════════════════════════════════════════════════════

def train_gpu_lstm(X_train, y_train, X_test, y_test, scaler):
    """LSTM trained on GPU with AMP. Lookback window → next day prediction."""
    logger.info("=" * 60)
    logger.info(f"GPU LSTM (device={config.DEVICE})")
    logger.info("=" * 60)

    X_t = torch.from_numpy(X_train).to(config.DEVICE)
    y_t = torch.from_numpy(y_train).to(config.DEVICE)

    # Simple LSTM model
    model = nn.Sequential(
        LSTMBlock(1, config.LSTM_HIDDEN, config.LSTM_LAYERS),
        nn.Linear(config.LSTM_HIDDEN, 1)
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LSTM_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    use_amp = config.DEVICE == "cuda"
    grad_scaler = GradScaler(enabled=use_amp)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.LSTM_BATCH, shuffle=True, drop_last=True)

    # Validation: last 15% of training data
    n_val = max(1, int(len(X_t) * 0.15))
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]

    best_loss, patience, best_state = float('inf'), 0, None
    history = {"train": [], "val": []}

    t0 = time.time()
    for epoch in range(config.LSTM_EPOCHS):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=config.DEVICE, enabled=use_amp):
                loss = criterion(model(xb).squeeze(-1), yb)
            if use_amp:
                grad_scaler.scale(loss).backward(); grad_scaler.step(optimizer); grad_scaler.update()
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
            logger.info(f"  Epoch {epoch+1:>3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        if patience >= 10:
            logger.info(f"  Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state); model = model.to(config.DEVICE)

    logger.info(f"  Done in {time.time()-t0:.1f}s | Best val MSE: {best_loss:.6f}")

    # Generate test predictions (inverse-transform back to original scale)
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(config.DEVICE)
        with autocast(device_type=config.DEVICE, enabled=use_amp):
            preds_scaled = model(X_test_t).squeeze(-1).cpu().numpy()

    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel().astype(np.float32)
    preds = np.clip(preds, 100, None)

    _plot_lstm_history(history)
    return model, preds


class LSTMBlock(nn.Module):
    """Minimal LSTM wrapper that returns only the last hidden state."""
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]  # last timestep only


def _plot_lstm_history(history):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["train"], '-', color='#FF7043', lw=2, label='Train MSE')
    ax.plot(history["val"], '-', color='#4FC3F7', lw=2, label='Val MSE')
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (scaled)")
    ax.set_title("🧠 LSTM Training Curves", fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_lstm_training.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_lstm_training.png")


def save_models(arima_model, ets_model, lstm_model):
    if arima_model:
        joblib.dump(arima_model, f"{config.MODEL_DIR}/day13_arima.joblib", compress=3)
    if ets_model:
        joblib.dump(ets_model, f"{config.MODEL_DIR}/day13_ets.joblib", compress=3)
    torch.save(lstm_model.state_dict(), f"{config.MODEL_DIR}/day13_lstm.pth")
    logger.info("  Models saved")
