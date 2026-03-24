"""
=============================================================================
 Day 8: Anemia Detection — Data Pipeline
=============================================================================
 KEY LEARNING: 
   1. Outlier removal techniques (IQR, Z-Score, Isolation Forest)
   2. Scaling strategies (Standard, MinMax, Robust)
   3. Impact of each on model performance
   
 WHY THIS MATTERS FOR BLOOD DATA:
   - Lab errors produce extreme outliers (e.g., Hemoglobin = 0 or 25)
   - Features have wildly different scales (RBC ~4.5M vs Hemoglobin ~13)
   - AdaBoost with Decision Trees is less scale-sensitive, BUT weak learners
     with extreme outliers still degrade the ensemble
=============================================================================
"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import config

logger = logging.getLogger(__name__)


def load_data():
    logger.info("=" * 60)
    logger.info("LOADING ANEMIA DETECTION DATASET")
    logger.info("=" * 60)
    try:
        df = pd.read_csv(f"{config.DATA_DIR}/anemia.csv")
        logger.info("Loaded from local CSV")
    except:
        logger.info("Generating realistic anemia blood test dataset...")
        df = _generate_fallback_data()
    logger.info(f"Dataset shape: {df.shape}")
    return df


def _generate_fallback_data():
    """
    Realistic CBC (Complete Blood Count) data for anemia detection.
    
    Anemia = low hemoglobin / low RBC. Key diagnostic thresholds:
      - Hemoglobin < 12 g/dL (women) or < 13 g/dL (men) = anemic
      - MCV < 80 fL = microcytic, > 100 fL = macrocytic
      - MCH < 27 pg = hypochromic
    """
    np.random.seed(config.RANDOM_SEED)
    n = 1421
    n_anemic = int(n * 0.38)  # ~38% anemic (realistic clinical prevalence in developing countries)
    n_healthy = n - n_anemic

    # Gender: 0=Male, 1=Female (anemia more common in females)
    gender_h = np.random.choice([0, 1], n_healthy, p=[0.55, 0.45])
    gender_a = np.random.choice([0, 1], n_anemic, p=[0.35, 0.65])

    # Hemoglobin (g/dL) — PRIMARY diagnostic marker
    hgb_h = np.random.normal(14.2, 1.5, n_healthy).clip(11.5, 18.0)
    hgb_a = np.random.normal(9.8, 2.2, n_anemic).clip(3.0, 13.5)

    # MCH — Mean Corpuscular Hemoglobin (pg)
    mch_h = np.random.normal(29.5, 2.0, n_healthy).clip(24, 35)
    mch_a = np.random.normal(24.0, 4.0, n_anemic).clip(12, 33)

    # MCHC — Mean Corpuscular Hemoglobin Concentration (g/dL)
    mchc_h = np.random.normal(33.5, 1.2, n_healthy).clip(30, 37)
    mchc_a = np.random.normal(30.5, 2.5, n_anemic).clip(22, 36)

    # MCV — Mean Corpuscular Volume (fL)
    mcv_h = np.random.normal(88, 5, n_healthy).clip(75, 100)
    mcv_a = np.random.normal(76, 12, n_anemic).clip(50, 110)

    # RBC count (millions/μL)
    rbc_h = np.random.normal(4.8, 0.5, n_healthy).clip(3.8, 6.2)
    rbc_a = np.random.normal(3.8, 0.8, n_anemic).clip(1.5, 5.5)

    # WBC count (thousands/μL) — less diagnostic for anemia
    wbc_h = np.random.normal(7.0, 2.0, n_healthy).clip(3.5, 15)
    wbc_a = np.random.normal(7.5, 2.8, n_anemic).clip(2.5, 18)

    # Platelet count (thousands/μL)
    plt_h = np.random.normal(250, 60, n_healthy).clip(150, 450)
    plt_a = np.random.normal(280, 90, n_anemic).clip(80, 600)

    data = {
        "Gender": np.concatenate([gender_h, gender_a]),
        "Hemoglobin": np.concatenate([hgb_h, hgb_a]),
        "MCH": np.concatenate([mch_h, mch_a]),
        "MCHC": np.concatenate([mchc_h, mchc_a]),
        "MCV": np.concatenate([mcv_h, mcv_a]),
        "RBC_count": np.concatenate([rbc_h, rbc_a]),
        "WBC_count": np.concatenate([wbc_h, wbc_a]),
        "Platelet_count": np.concatenate([plt_h, plt_a]),
        "Result": np.concatenate([np.zeros(n_healthy), np.ones(n_anemic)]).astype(int)
    }
    df = pd.DataFrame(data)

    # Inject realistic outliers (~3% lab errors / extreme cases)
    np.random.seed(config.RANDOM_SEED + 1)
    n_outliers = int(n * 0.03)
    outlier_idx = np.random.choice(n, n_outliers, replace=False)
    for idx in outlier_idx:
        feat = np.random.choice(["Hemoglobin", "MCH", "MCV", "RBC_count", "WBC_count", "Platelet_count"])
        if feat == "Hemoglobin":
            df.loc[idx, feat] = np.random.choice([2.0, 2.5, 19.0, 20.5])
        elif feat == "MCH":
            df.loc[idx, feat] = np.random.choice([8.0, 10.0, 42.0, 45.0])
        elif feat == "MCV":
            df.loc[idx, feat] = np.random.choice([40.0, 45.0, 115.0, 125.0])
        elif feat == "RBC_count":
            df.loc[idx, feat] = np.random.choice([1.0, 1.2, 7.5, 8.0])
        elif feat == "WBC_count":
            df.loc[idx, feat] = np.random.choice([0.5, 1.0, 25.0, 35.0])
        elif feat == "Platelet_count":
            df.loc[idx, feat] = np.random.choice([20.0, 30.0, 700.0, 900.0])

    # 2% label noise
    noise_idx = np.random.choice(n, int(n * 0.02), replace=False)
    df.loc[noise_idx, "Result"] = 1 - df.loc[noise_idx, "Result"]

    return df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)


def explore_data(df):
    logger.info("-" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("-" * 60)
    logger.info(f"Samples: {len(df)} | Features: {len(config.FEATURE_NAMES)}")
    logger.info(f"Missing: {df.isnull().sum().sum()}")

    cd = df[config.TARGET_NAME].value_counts()
    for cls, cnt in cd.items():
        label = config.CLASS_NAMES[int(cls)]
        logger.info(f"  {label}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # Stats
    logger.info(f"\nFeature Statistics:")
    for feat in config.FEATURE_NAMES[1:]:  # skip Gender
        vals = df[feat]
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        outliers = ((vals < q1 - 1.5*iqr) | (vals > q3 + 1.5*iqr)).sum()
        logger.info(f"  {feat:18s}: mean={vals.mean():8.2f} | std={vals.std():7.2f} | "
                     f"range=[{vals.min():.1f}, {vals.max():.1f}] | outliers={outliers}")

    # ─── Plot 1: Class + Feature Distributions ──────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    # Class dist
    colors_class = ["#66BB6A", "#EF5350"]
    bars = axes[0].bar(config.CLASS_NAMES, cd.values, color=colors_class, edgecolor="white")
    for bar, cnt in zip(bars, cd.values):
        axes[0].text(bar.get_x()+bar.get_width()/2, cnt+5, f'{cnt}\n({cnt/len(df)*100:.1f}%)',
                     ha='center', fontweight='bold', fontsize=9)
    axes[0].set_title("🩸 Class Distribution", fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)

    # Feature distributions with outlier markers
    for i, feat in enumerate(config.FEATURE_NAMES[1:]):
        ax = axes[i+1]
        for val, color, label in [(0, "#4FC3F7", "Healthy"), (1, "#EF5350", "Anemic")]:
            subset = df[df["Result"] == val][feat]
            ax.hist(subset, bins=25, alpha=0.6, color=color, label=label, edgecolor='white')
        ax.set_title(f"🔬 {feat}", fontweight='bold', fontsize=10)
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle("Blood Test Feature Distributions — Anemic vs Healthy", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_eda_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_eda_distributions.png")

    # ─── Plot 2: Boxplots showing outliers ──────────────────────────
    fig, axes = plt.subplots(1, 7, figsize=(24, 5))
    for i, feat in enumerate(config.FEATURE_NAMES[1:]):
        bp = df.boxplot(column=feat, by="Result", ax=axes[i], patch_artist=True,
                        boxprops=dict(facecolor='#4FC3F7', alpha=0.6),
                        medianprops=dict(color='#EF5350', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
        axes[i].set_title(feat.replace('_', '\n'), fontsize=9, fontweight='bold')
        axes[i].set_xlabel("")
    fig.suptitle("🔍 Boxplots — Outliers Visible (Red Dots)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_boxplots_outliers.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_boxplots_outliers.png")

    return cd


def detect_outliers(X, y, method="iqr"):
    """
    Detect and remove outliers using specified method.
    
    METHODS COMPARED:
    1. IQR (Interquartile Range): Remove if < Q1-1.5*IQR or > Q3+1.5*IQR
       - Simple, robust, widely used in clinical labs
       - Can be too aggressive on non-normal distributions
       
    2. Z-Score: Remove if |z| > 3 (3 standard deviations from mean)
       - Assumes normal distribution
       - Less robust than IQR for skewed data
       
    3. Isolation Forest: ML-based anomaly detection
       - No distribution assumptions
       - Works well in high dimensions
       - Can capture multivariate outliers that IQR/Z-score miss
    """
    n_before = len(X)

    if method == "none":
        return X, y, 0

    elif method == "iqr":
        mask = np.ones(len(X), dtype=bool)
        for col in range(X.shape[1]):
            q1 = np.percentile(X[:, col], 25)
            q3 = np.percentile(X[:, col], 75)
            iqr = q3 - q1
            lower = q1 - config.IQR_MULTIPLIER * iqr
            upper = q3 + config.IQR_MULTIPLIER * iqr
            mask &= (X[:, col] >= lower) & (X[:, col] <= upper)
        X_clean, y_clean = X[mask], y[mask]

    elif method == "zscore":
        from scipy import stats
        z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
        mask = (z_scores < config.ZSCORE_THRESHOLD).all(axis=1)
        X_clean, y_clean = X[mask], y[mask]

    elif method == "isolation_forest":
        iso = IsolationForest(contamination=0.05, random_state=config.RANDOM_SEED, n_jobs=-1)
        preds = iso.fit_predict(X)
        mask = preds == 1
        X_clean, y_clean = X[mask], y[mask]

    else:
        return X, y, 0

    n_removed = n_before - len(X_clean)
    return X_clean, y_clean, n_removed


def get_scaler(method="standard"):
    """Return scaler based on method name."""
    if method == "standard":
        return StandardScaler()
    elif method == "minmax":
        return MinMaxScaler()
    elif method == "robust":
        return RobustScaler()  # Uses median/IQR instead of mean/std — robust to outliers!
    else:
        return None


def preprocess_and_split(df):
    logger.info("-" * 60)
    logger.info("PREPROCESSING & SPLITTING")
    logger.info("-" * 60)
    df = df.copy()

    X = df[config.FEATURE_NAMES].values.astype(np.float64)
    y = df[config.TARGET_NAME].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y)

    logger.info(f"Split: Train={len(y_train)} | Test={len(y_test)}")
    logger.info(f"  Train anemic rate: {y_train.mean():.3f}")
    logger.info(f"  Test anemic rate:  {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test
