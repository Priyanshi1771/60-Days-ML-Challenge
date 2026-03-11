"""Day 11: ICU Mortality — Data Pipeline"""
import logging, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config

logger = logging.getLogger(__name__)


def load_data():
    """Load or generate realistic ICU patient data."""
    logger.info("=" * 60)
    logger.info("LOADING ICU MORTALITY DATASET")
    logger.info("=" * 60)

    rng = np.random.RandomState(config.RANDOM_SEED)
    n = 4200

    # Generate correlated ICU features
    age = rng.normal(62, 16, n).clip(18, 95).astype(np.float32)
    heart_rate = (rng.normal(88, 18, n) + (age - 60) * 0.3).clip(40, 180).astype(np.float32)
    sys_bp = (rng.normal(120, 25, n) - (age - 60) * 0.4).clip(60, 220).astype(np.float32)
    dia_bp = (sys_bp * 0.6 + rng.normal(0, 8, n)).clip(30, 130).astype(np.float32)
    resp_rate = (rng.normal(18, 5, n) + (age - 60) * 0.05).clip(8, 45).astype(np.float32)
    temperature = rng.normal(37.0, 0.7, n).clip(34, 41).astype(np.float32)
    spo2 = (rng.normal(96, 3, n) - (age - 60) * 0.03).clip(70, 100).astype(np.float32)
    gcs = rng.choice(np.arange(3, 16), n, p=np.array([3,2,2,2,3,3,4,5,8,12,16,20,20])/100).astype(np.float32)
    bun = (rng.normal(22, 14, n) + (age - 60) * 0.3).clip(5, 120).astype(np.float32)
    creatinine = (rng.lognormal(0.2, 0.6, n) + (age - 60) * 0.01).clip(0.3, 12).astype(np.float32)
    sodium = rng.normal(139, 4.5, n).clip(120, 160).astype(np.float32)
    potassium = rng.normal(4.2, 0.7, n).clip(2.5, 7.5).astype(np.float32)
    hemoglobin = (rng.normal(11.5, 2.2, n) - (age - 60) * 0.02).clip(4, 18).astype(np.float32)
    wbc = rng.lognormal(2.2, 0.5, n).clip(1, 50).astype(np.float32)
    platelet = rng.normal(210, 85, n).clip(10, 600).astype(np.float32)
    lactate = rng.lognormal(0.5, 0.7, n).clip(0.3, 20).astype(np.float32)
    pf_ratio = (rng.normal(300, 100, n) - lactate * 15).clip(50, 500).astype(np.float32)
    urine = (rng.normal(1200, 500, n) - (age - 60) * 5).clip(0, 4000).astype(np.float32)
    ventilator = rng.binomial(1, 0.35, n).astype(np.float32)
    vasopressor = rng.binomial(1, 0.25, n).astype(np.float32)
    prev_icu = rng.poisson(0.4, n).clip(0, 5).astype(np.float32)
    los_before = rng.exponential(2.5, n).clip(0, 30).astype(np.float32)

    X = np.column_stack([age, heart_rate, sys_bp, dia_bp, resp_rate, temperature,
                          spo2, gcs, bun, creatinine, sodium, potassium, hemoglobin,
                          wbc, platelet, lactate, pf_ratio, urine, ventilator,
                          vasopressor, prev_icu, los_before])

    # Target: mortality risk (0-1) — nonlinear combination with noise
    y = (
        0.008 * age
        + 0.003 * heart_rate
        - 0.002 * sys_bp
        + 0.005 * resp_rate
        - 0.01 * spo2
        - 0.025 * gcs
        + 0.004 * bun
        + 0.03 * creatinine
        + 0.04 * lactate
        - 0.0005 * pf_ratio
        - 0.00008 * urine
        + 0.08 * ventilator
        + 0.12 * vasopressor
        + 0.03 * prev_icu
        + 0.0001 * (age * lactate)         # interaction term
        - 0.0002 * (gcs * pf_ratio)        # interaction term
        + 0.00005 * age**2                  # polynomial effect
        + rng.normal(0, 0.06, n)            # noise
    ).astype(np.float32)

    # Clip and normalize to [0, 1]
    y = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))
    y = (y - y.min()) / (y.max() - y.min())
    y = y.astype(np.float32)

    logger.info(f"Generated {n} ICU patients | {X.shape[1]} features | Target range: [{y.min():.3f}, {y.max():.3f}]")
    return X, y


def explore_data(X, y):
    """Lightweight EDA."""
    logger.info("-" * 60)
    logger.info("EDA")
    logger.info("-" * 60)
    logger.info(f"Shape: {X.shape} | Target: mean={y.mean():.3f}, std={y.std():.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Target distribution
    axes[0,0].hist(y, bins=40, color='#EF5350', alpha=0.8, edgecolor='white')
    axes[0,0].set_title("🏥 Mortality Risk Distribution", fontweight='bold')
    axes[0,0].set_xlabel("Risk Score"); axes[0,0].spines[['top','right']].set_visible(False)

    # Top correlated features
    corrs = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    top_idx = np.argsort(np.abs(corrs))[::-1][:5]

    for ax_i, fi in enumerate(top_idx):
        ax = axes.ravel()[ax_i + 1]
        ax.scatter(X[:, fi], y, alpha=0.15, s=8, color='#4FC3F7', rasterized=True)
        ax.set_xlabel(config.FEATURE_NAMES[fi])
        ax.set_ylabel("Mortality Risk")
        r = corrs[fi]
        ax.set_title(f"🔬 {config.FEATURE_NAMES[fi]} (r={r:.3f})", fontweight='bold', fontsize=10)
        ax.spines[['top','right']].set_visible(False)

    plt.suptitle("ICU Mortality — Key Feature Correlations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_eda.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_eda.png")


def preprocess_and_split(X, y):
    """Split → scale (fit on train only). Returns float32."""
    logger.info("-" * 60)
    logger.info("PREPROCESSING")
    logger.info("-" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler
