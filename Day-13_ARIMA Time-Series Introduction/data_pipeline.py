"""Day 13: COVID-19 Forecasting — Data Pipeline"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import config

logger = logging.getLogger(__name__)


def load_data():
    logger.info("=" * 60)
    logger.info("LOADING COVID-19 TIME-SERIES DATA")
    logger.info("=" * 60)

    rng = np.random.RandomState(config.RANDOM_SEED)
    n = config.N_DAYS
    t = np.arange(n, dtype=np.float32)
    dates = pd.date_range("2020-03-01", periods=n, freq="D")

    # Simulate 3 COVID waves with different peaks
    wave1 = 8000 * np.exp(-0.5 * ((t - 80) / 25) ** 2)     # Spring 2020
    wave2 = 15000 * np.exp(-0.5 * ((t - 280) / 35) ** 2)    # Winter 2020-21
    wave3 = 20000 * np.exp(-0.5 * ((t - 500) / 40) ** 2)    # Winter 2021-22
    decline = 3000 * np.exp(-0.5 * ((t - 650) / 30) ** 2)   # Omicron tail

    trend = wave1 + wave2 + wave3 + decline

    # Weekly seasonality (fewer cases reported on weekends)
    weekly = -0.15 * trend * np.sin(2 * np.pi * t / 7)

    # Random noise (proportional to signal)
    noise = rng.normal(0, 1, n) * (trend * 0.08 + 200)

    cases = (trend + weekly + noise + 500).clip(100, None).astype(np.float32)

    df = pd.DataFrame({"date": dates, "cases": cases})
    df.set_index("date", inplace=True)

    logger.info(f"Generated {n} days | Range: [{cases.min():.0f}, {cases.max():.0f}]")
    logger.info(f"Period: {dates[0].date()} → {dates[-1].date()}")
    return df


def explore_data(df):
    logger.info("-" * 60)
    logger.info("EDA — TIME-SERIES ANALYSIS")
    logger.info("-" * 60)

    cases = df["cases"].values
    logger.info(f"Mean: {cases.mean():.0f} | Std: {cases.std():.0f}")
    logger.info(f"Min: {cases.min():.0f} | Max: {cases.max():.0f}")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 1.5, 1.5]})

    # Full time series
    axes[0].plot(df.index, cases, color='#EF5350', lw=1.2, alpha=0.9)
    axes[0].fill_between(df.index, cases, alpha=0.15, color='#EF5350')

    # Mark waves
    for peak, label, color in [(80, "Wave 1", "#FF7043"), (280, "Wave 2", "#AB47BC"),
                                 (500, "Wave 3", "#4FC3F7"), (650, "Decline", "#66BB6A")]:
        if peak < len(df):
            axes[0].axvline(df.index[peak], color=color, linestyle='--', alpha=0.5, lw=1.5)
            axes[0].text(df.index[peak], cases.max() * 0.95, f" {label}", fontsize=9, color=color)

    # Train/test split line
    split_idx = int(len(df) * config.TRAIN_RATIO)
    axes[0].axvline(df.index[split_idx], color='#FFB74D', lw=2.5, linestyle='-.',
                     label=f'Train/Test split ({config.TRAIN_RATIO*100:.0f}%/{(1-config.TRAIN_RATIO)*100:.0f}%)')
    axes[0].set_title("🦠 COVID-19 Daily New Cases — Full Timeline", fontweight='bold', fontsize=13)
    axes[0].set_ylabel("Daily Cases")
    axes[0].legend(fontsize=10)
    axes[0].spines[['top', 'right']].set_visible(False)

    # 7-day rolling average
    rolling = pd.Series(cases).rolling(7).mean()
    axes[1].plot(df.index, cases, alpha=0.3, color='#EF5350', lw=0.8, label='Raw')
    axes[1].plot(df.index, rolling, color='#4FC3F7', lw=2.5, label='7-day MA')
    axes[1].set_title("📊 7-Day Moving Average (smooths weekly noise)", fontweight='bold')
    axes[1].legend(fontsize=9); axes[1].spines[['top', 'right']].set_visible(False)

    # Weekly pattern (last 8 weeks)
    last_56 = cases[-56:]
    weekly_avg = [last_56[i::7].mean() for i in range(7)]
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    colors = ['#4FC3F7'] * 5 + ['#FF7043'] * 2
    axes[2].bar(days, weekly_avg, color=colors, edgecolor='white')
    axes[2].set_title("📅 Weekly Seasonality (weekends = fewer reports)", fontweight='bold')
    axes[2].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_eda_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_eda_timeseries.png")


def split_data(df):
    """Temporal split — NO shuffle (time-series must preserve order)."""
    logger.info("-" * 60)
    logger.info("TEMPORAL TRAIN/TEST SPLIT")
    logger.info("-" * 60)

    split_idx = int(len(df) * config.TRAIN_RATIO)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    logger.info(f"Train: {len(train)} days ({train.index[0].date()} → {train.index[-1].date()})")
    logger.info(f"Test:  {len(test)} days ({test.index[0].date()} → {test.index[-1].date()})")
    return train, test


def prepare_lstm_data(train_series, test_series, lookback=config.LSTM_LOOKBACK):
    """Create sliding window sequences for LSTM. Returns float32 tensors."""
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).astype(np.float32).ravel()
    test_scaled = scaler.transform(test_series.values.reshape(-1, 1)).astype(np.float32).ravel()

    full_scaled = np.concatenate([train_scaled, test_scaled])

    def _windows(data, lb):
        X, y = [], []
        for i in range(lb, len(data)):
            X.append(data[i - lb:i])
            y.append(data[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # Train windows from train portion only
    X_train, y_train = _windows(train_scaled, lookback)

    # Test windows: need last `lookback` train values as prefix
    test_with_prefix = np.concatenate([train_scaled[-lookback:], test_scaled])
    X_test, y_test = _windows(test_with_prefix, lookback)

    # Reshape for LSTM: (batch, seq_len, features=1)
    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]

    logger.info(f"LSTM data: Train={X_train.shape} | Test={X_test.shape} | Lookback={lookback}")
    return X_train, y_train, X_test, y_test, scaler
