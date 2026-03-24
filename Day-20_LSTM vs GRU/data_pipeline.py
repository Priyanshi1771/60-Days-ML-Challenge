"""Day 20: Viral Load Forecasting — Data Pipeline"""
import logging, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import config

logger = logging.getLogger(__name__)


def load_data():
    logger.info("=" * 60)
    logger.info("LOADING HIV VIRAL LOAD TIME-SERIES")
    logger.info("=" * 60)

    rng = np.random.RandomState(config.RANDOM_SEED)
    n_pat = config.N_PATIENTS
    T = config.TIMEPOINTS
    all_series = []

    for p in range(n_pat):
        initial = rng.uniform(3.5, 6.0)  # log10 copies/mL
        decay_rate = rng.uniform(0.03, 0.12)
        t = np.arange(T, dtype=np.float32)
        vl = initial * np.exp(-decay_rate * t)

        # 30% of patients experience viral rebound
        if rng.random() < 0.3:
            rb_time = rng.randint(20, 45)
            vl += 2.0 * np.exp(-0.05 * (t - rb_time) ** 2) * (t > rb_time)

        vl += rng.normal(0, 0.15, T)
        all_series.append(np.clip(vl, 1.0, 7.0).astype(np.float32))

    data = np.array(all_series)
    logger.info(f"Generated {n_pat} patients x {T} weeks")
    logger.info(f"Viral load (log10): mean={data.mean():.2f}, range=[{data.min():.1f}, {data.max():.1f}]")
    return data


def explore_data(data):
    logger.info("-" * 60)
    logger.info("EDA")

    rng = np.random.RandomState(config.RANDOM_SEED)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sample trajectories
    for i in rng.choice(data.shape[0], 15, replace=False):
        axes[0].plot(data[i], alpha=0.5, lw=1.2)
    axes[0].set_title("Sample Patient Trajectories", fontweight='bold')
    axes[0].set_xlabel("Week"); axes[0].set_ylabel("log10(copies/mL)")
    axes[0].spines[['top', 'right']].set_visible(False)

    # Mean + CI
    mean_vl = data.mean(axis=0)
    std_vl = data.std(axis=0)
    weeks = np.arange(data.shape[1])
    axes[1].plot(weeks, mean_vl, color='#EF5350', lw=2.5)
    axes[1].fill_between(weeks, mean_vl - std_vl, mean_vl + std_vl, alpha=0.2, color='#EF5350')
    axes[1].set_title("Mean Trajectory (+/- 1 SD)", fontweight='bold')
    axes[1].set_xlabel("Week"); axes[1].set_ylabel("log10(copies/mL)")
    axes[1].spines[['top', 'right']].set_visible(False)

    # Distributions at timepoints
    for w, c in [(0, '#EF5350'), (12, '#FF7043'), (26, '#FFB74D'), (51, '#66BB6A')]:
        axes[2].hist(data[:, w], bins=20, alpha=0.5, color=c, label=f'Week {w}', edgecolor='white')
    axes[2].set_title("Distribution by Week", fontweight='bold')
    axes[2].set_xlabel("log10(copies/mL)"); axes[2].legend()
    axes[2].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_eda.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_eda.png")


def prepare_sequences(data):
    """Sliding window sequences for LSTM/GRU."""
    logger.info("-" * 60)
    logger.info("PREPARING SEQUENCES")

    scaler = MinMaxScaler()
    flat = data.reshape(-1, 1)
    scaler.fit(flat)
    data_scaled = scaler.transform(flat).reshape(data.shape).astype(np.float32)

    lb = config.LOOKBACK
    X_all, y_all = [], []
    for patient in data_scaled:
        for i in range(lb, len(patient)):
            X_all.append(patient[i - lb:i])
            y_all.append(patient[i])

    X = np.array(X_all, dtype=np.float32)[:, :, np.newaxis]
    y = np.array(y_all, dtype=np.float32)

    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    logger.info(f"Sequences: Train={X_train.shape[0]} | Test={X_test.shape[0]} | Lookback={lb}")
    return X_train, X_test, y_train, y_test, scaler
