"""
=============================================================================
 Day 6: Kidney Disease Prediction — Data Pipeline
=============================================================================
 Handles: Loading UCI CKD dataset, extensive missing value imputation,
          mixed-type encoding, preprocessing, stratified splitting
 
 KEY CHALLENGE: This dataset has ~35% missing values across mixed types
 (numeric + categorical). Proper imputation strategy is critical.
 
 KEY PRINCIPLE: All imputation fitted on train set ONLY.
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

import config

logger = logging.getLogger(__name__)


def load_data():
    """
    Load the UCI Chronic Kidney Disease dataset.

    Dataset characteristics:
    - 400 instances, 24 features + 1 target
    - 11 numeric, 13 nominal features
    - ~35% missing values (realistic clinical scenario!)
    - Binary classification: CKD (250) vs Not-CKD (150)
    - Class imbalance ratio: ~1.67:1
    """
    logger.info("=" * 60)
    logger.info("LOADING UCI CHRONIC KIDNEY DISEASE DATASET")
    logger.info("=" * 60)

    try:
        from sklearn.datasets import fetch_openml
        ckd = fetch_openml(data_id=4875, as_frame=True, parser="auto")
        df = ckd.data.copy()
        df[config.TARGET_NAME] = ckd.target
        logger.info("Dataset loaded from OpenML")
    except Exception as e:
        logger.warning(f"OpenML fetch failed: {e}")
        logger.info("Generating synthetic CKD-like dataset as fallback...")
        df = _generate_fallback_data()

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features: {df.shape[1] - 1}")
    logger.info(f"Target: {config.TARGET_NAME}")

    return df


def _generate_fallback_data():
    """Generate synthetic CKD data with realistic class overlap (~95-98% accuracy)."""
    np.random.seed(config.RANDOM_SEED)
    n = 400
    n_ckd, n_notckd = 250, 150

    data = {}

    # ── Numeric features — REDUCED separation between classes ────────
    # Age: significant overlap (CKD skews older but not dramatically)
    data["age"] = np.concatenate([
        np.random.normal(52, 16, n_ckd).clip(5, 90),
        np.random.normal(46, 16, n_notckd).clip(5, 90)
    ])
    # Blood pressure: very similar distributions
    data["bp"] = np.concatenate([
        np.random.normal(78, 12, n_ckd).clip(50, 120),
        np.random.normal(74, 10, n_notckd).clip(50, 110)
    ])
    # Specific gravity: overlapping discrete values
    data["sg"] = np.concatenate([
        np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025],
                         n_ckd, p=[0.22, 0.30, 0.25, 0.15, 0.08]),
        np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025],
                         n_notckd, p=[0.08, 0.15, 0.25, 0.30, 0.22])
    ])
    # Albumin: moderate overlap
    data["al"] = np.concatenate([
        np.random.choice([0, 1, 2, 3, 4, 5], n_ckd, p=[
                         0.25, 0.22, 0.22, 0.16, 0.10, 0.05]),
        np.random.choice([0, 1, 2, 3, 4, 5], n_notckd, p=[
                         0.55, 0.22, 0.13, 0.06, 0.03, 0.01])
    ])
    # Sugar: heavy overlap
    data["su"] = np.concatenate([
        np.random.choice([0, 1, 2, 3, 4, 5], n_ckd, p=[
                         0.40, 0.25, 0.15, 0.12, 0.05, 0.03]),
        np.random.choice([0, 1, 2, 3, 4, 5], n_notckd, p=[
                         0.65, 0.18, 0.10, 0.04, 0.02, 0.01])
    ])
    # Blood glucose: closer means, higher std
    data["bgr"] = np.concatenate([
        np.random.normal(145, 55, n_ckd).clip(60, 490),
        np.random.normal(115, 35, n_notckd).clip(60, 300)
    ])
    # Blood urea: moderate separation
    data["bu"] = np.concatenate([
        np.random.normal(55, 30, n_ckd).clip(10, 200),
        np.random.normal(35, 15, n_notckd).clip(10, 80)
    ])
    # Serum creatinine: KEY discriminator but with overlap
    data["sc"] = np.concatenate([
        np.random.normal(3.0, 2.5, n_ckd).clip(0.4, 25),
        np.random.normal(1.2, 0.6, n_notckd).clip(0.4, 4.0)
    ])
    # Sodium: very similar (hard to distinguish)
    data["sod"] = np.concatenate([
        np.random.normal(133, 7, n_ckd).clip(100, 155),
        np.random.normal(139, 5, n_notckd).clip(115, 155)
    ])
    # Potassium: almost identical distributions
    data["pot"] = np.concatenate([
        np.random.normal(4.6, 1.0, n_ckd).clip(2.5, 10),
        np.random.normal(4.3, 0.7, n_notckd).clip(2.5, 7)
    ])
    # Hemoglobin: moderate separation
    data["hemo"] = np.concatenate([
        np.random.normal(11.0, 2.5, n_ckd).clip(3, 17),
        np.random.normal(14.0, 2.0, n_notckd).clip(8, 18)
    ])
    # PCV: correlated with hemoglobin, overlap zone
    data["pcv"] = np.concatenate([
        np.random.normal(34, 8, n_ckd).clip(10, 55),
        np.random.normal(42, 6, n_notckd).clip(20, 55)
    ])
    # WBC: very noisy, minimal class signal
    data["wc"] = np.concatenate([
        np.random.normal(8500, 3000, n_ckd).clip(3000, 25000),
        np.random.normal(7800, 2200, n_notckd).clip(3000, 18000)
    ])
    # RBC: moderate overlap
    data["rc"] = np.concatenate([
        np.random.normal(4.0, 1.0, n_ckd).clip(2, 7),
        np.random.normal(4.8, 0.8, n_notckd).clip(2.5, 7)
    ])

    # ── Categorical features — MORE noise / overlap ──────────────────
    data["rbc"] = np.concatenate([
        np.random.choice(["normal", "abnormal"], n_ckd, p=[0.45, 0.55]),
        np.random.choice(["normal", "abnormal"], n_notckd, p=[0.78, 0.22])
    ])
    data["pc"] = np.concatenate([
        np.random.choice(["normal", "abnormal"], n_ckd, p=[0.42, 0.58]),
        np.random.choice(["normal", "abnormal"], n_notckd, p=[0.82, 0.18])
    ])
    data["pcc"] = np.concatenate([
        np.random.choice(["present", "notpresent"], n_ckd, p=[0.28, 0.72]),
        np.random.choice(["present", "notpresent"], n_notckd, p=[0.06, 0.94])
    ])
    data["ba"] = np.concatenate([
        np.random.choice(["present", "notpresent"], n_ckd, p=[0.15, 0.85]),
        np.random.choice(["present", "notpresent"], n_notckd, p=[0.04, 0.96])
    ])
    data["htn"] = np.concatenate([
        np.random.choice(["yes", "no"], n_ckd, p=[0.55, 0.45]),
        np.random.choice(["yes", "no"], n_notckd, p=[0.18, 0.82])
    ])
    data["dm"] = np.concatenate([
        np.random.choice(["yes", "no"], n_ckd, p=[0.42, 0.58]),
        np.random.choice(["yes", "no"], n_notckd, p=[0.12, 0.88])
    ])
    data["cad"] = np.concatenate([
        np.random.choice(["yes", "no"], n_ckd, p=[0.16, 0.84]),
        np.random.choice(["yes", "no"], n_notckd, p=[0.05, 0.95])
    ])
    data["appet"] = np.concatenate([
        np.random.choice(["good", "poor"], n_ckd, p=[0.50, 0.50]),
        np.random.choice(["good", "poor"], n_notckd, p=[0.82, 0.18])
    ])
    data["pe"] = np.concatenate([
        np.random.choice(["yes", "no"], n_ckd, p=[0.38, 0.62]),
        np.random.choice(["yes", "no"], n_notckd, p=[0.10, 0.90])
    ])
    data["ane"] = np.concatenate([
        np.random.choice(["yes", "no"], n_ckd, p=[0.35, 0.65]),
        np.random.choice(["yes", "no"], n_notckd, p=[0.10, 0.90])
    ])

    # ── Target ───────────────────────────────────────────────────────
    data[config.TARGET_NAME] = ["ckd"] * n_ckd + ["notckd"] * n_notckd

    df = pd.DataFrame(data)

    # ── Inject ~25% missing values (higher = harder) ─────────────────
    np.random.seed(config.RANDOM_SEED + 1)
    for col in df.columns:
        if col != config.TARGET_NAME:
            miss_rate = np.random.uniform(0.12, 0.30)  # 12-30% per column
            mask = np.random.random(n) < miss_rate
            df.loc[mask, col] = np.nan

    # ── Inject ~5% label noise (swap some CKD ↔ Not-CKD) ────────────
    np.random.seed(config.RANDOM_SEED + 2)
    noise_idx = np.random.choice(n, size=int(n * 0.04), replace=False)
    for idx in noise_idx:
        df.iloc[idx, -1] = "notckd" if df.iloc[idx, -1] == "ckd" else "ckd"

    return df


def explore_data(df):
    """
    Comprehensive EDA with focus on missing values (key challenge for CKD).
    """
    logger.info("-" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("-" * 60)

    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Total cells: {df.size}")
    logger.info(
        f"Total missing: {df.isnull().sum().sum()} ({df.isnull().sum().sum()/df.size*100:.1f}%)")

    # Class distribution
    class_dist = df[config.TARGET_NAME].value_counts()
    logger.info(f"\nClass Distribution:")
    for cls, count in class_dist.items():
        logger.info(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")

    # Missing values per feature
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(1)
    logger.info(f"\nMissing Values (top 10):")
    for col in missing.head(10).index:
        if missing[col] > 0:
            logger.info(
                f"  {col:25s}: {missing[col]:4d} ({missing_pct[col]:5.1f}%)")

    # ─── Plot 1: Class Distribution ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#EF5350", "#66BB6A"]
    bars = axes[0].bar(config.CLASS_NAMES, class_dist.values,
                       color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, class_dist.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+2,
                     f'{count}\n({count/len(df)*100:.1f}%)', ha='center', fontweight='bold')
    axes[0].set_title("Kidney Disease — Class Distribution",
                      fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Count")
    axes[0].spines[['top', 'right']].set_visible(False)

    # ─── Plot 2: Missing Values Heatmap ──────────────────────────────────
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        missing_data = df[missing_cols.index].isnull().astype(int)
        sns.heatmap(missing_data.T, cbar_kws={'label': 'Missing'}, cmap="YlOrRd",
                    ax=axes[1], yticklabels=True)
        axes[1].set_title("Missing Value Pattern",
                          fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Samples")
    else:
        axes[1].text(0.5, 0.5, "No missing values",
                     ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_class_and_missing.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_class_and_missing.png")

    # ─── Plot 3: Feature distributions (numeric) ────────────────────────
    numeric_cols = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    n_cols = min(len(numeric_cols), 14)
    if n_cols > 0:
        rows = (n_cols + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(20, rows * 4))
        axes = axes.ravel() if rows > 1 else [
            axes] if rows == 1 and n_cols == 1 else axes.ravel()

        for i, col in enumerate(numeric_cols[:n_cols]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].hist(ax=axes[i], bins=20, color='#4FC3F7',
                         alpha=0.7, edgecolor='white')
            axes[i].set_title(col, fontsize=10, fontweight='bold')
            axes[i].spines[['top', 'right']].set_visible(False)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Numeric Feature Distributions",
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f"{config.PLOT_DIR}/02_feature_distributions.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved: 02_feature_distributions.png")

    return class_dist


def preprocess_and_split(df):
    """
    Heavy preprocessing for CKD dataset.

    Steps:
    1. Clean target labels
    2. Convert numeric columns to float
    3. Encode categorical features (LabelEncoder per column)
    4. Stratified train/test split
    5. Impute missing values (median for numeric, mode for categorical) 
       — FIT ON TRAIN ONLY
    6. Scale features — FIT ON TRAIN ONLY

    WHY THIS ORDER MATTERS:
    - Split BEFORE imputing to prevent data leakage
    - Imputation statistics (median, mode) from test data could leak info
    - Same principle as scaling: learn parameters from train only
    """
    logger.info("-" * 60)
    logger.info("PREPROCESSING & SPLITTING")
    logger.info("-" * 60)

    df = df.copy()

    # ── 1. Clean target ──────────────────────────────────────────────────
    df[config.TARGET_NAME] = df[config.TARGET_NAME].astype(
        str).str.strip().str.lower()
    df[config.TARGET_NAME] = df[config.TARGET_NAME].map(
        lambda x: 1 if 'ckd' == x or x == 'ckd' else 0 if 'notckd' in x or 'not' in x else None
    )
    # Handle edge cases
    if df[config.TARGET_NAME].isnull().any():
        # Try numeric mapping
        df[config.TARGET_NAME] = df[config.TARGET_NAME].fillna(
            df[config.TARGET_NAME].mode()[0] if len(
                df[config.TARGET_NAME].mode()) > 0 else 1
        )
    df[config.TARGET_NAME] = df[config.TARGET_NAME].astype(int)

    logger.info(f"Target encoded: 1=CKD, 0=Not-CKD")
    logger.info(
        f"  CKD: {(df[config.TARGET_NAME] == 1).sum()}, Not-CKD: {(df[config.TARGET_NAME] == 0).sum()}")

    # ── 2. Convert numeric columns ───────────────────────────────────────
    for col in config.NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── 3. Encode categorical features ───────────────────────────────────
    label_encoders = {}
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            non_null = df[col].dropna().astype(str).str.strip().str.lower()
            le.fit(non_null.unique())
            label_encoders[col] = le
            # Build new column with encoded values (NaN preserved)
            new_col = pd.Series(np.nan, index=df.index, dtype=float)
            mask = df[col].notna()
            new_col[mask] = le.transform(df.loc[mask, col].astype(
                str).str.strip().str.lower()).astype(float)
            df[col] = new_col

    logger.info(f"Encoded {len(label_encoders)} categorical features")

    # ── 4. Separate features and target ──────────────────────────────────
    feature_cols = [c for c in config.NUMERIC_FEATURES +
                    config.CATEGORICAL_FEATURES if c in df.columns]
    X = df[feature_cols].values.astype(np.float64)
    y = df[config.TARGET_NAME].values.astype(int)

    # ── 5. Stratified split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y
    )

    logger.info(f"\nSplit: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    logger.info(f"  Train CKD ratio: {y_train.mean():.3f}")
    logger.info(f"  Test CKD ratio:  {y_test.mean():.3f}")

    # ── 6. Impute missing values (FIT ON TRAIN ONLY) ─────────────────────
    num_idx = list(range(len(config.NUMERIC_FEATURES)))
    cat_idx = list(range(len(config.NUMERIC_FEATURES), X_train.shape[1]))

    # Numeric: median imputation
    num_imputer = SimpleImputer(strategy='median')
    if len(num_idx) > 0 and max(num_idx) < X_train.shape[1]:
        X_train[:, num_idx] = num_imputer.fit_transform(X_train[:, num_idx])
        X_test[:, num_idx] = num_imputer.transform(X_test[:, num_idx])

    # Categorical: most frequent imputation
    cat_imputer = SimpleImputer(strategy='most_frequent')
    if len(cat_idx) > 0 and max(cat_idx) < X_train.shape[1]:
        X_train[:, cat_idx] = cat_imputer.fit_transform(X_train[:, cat_idx])
        X_test[:, cat_idx] = cat_imputer.transform(X_test[:, cat_idx])

    missing_train = np.isnan(X_train).sum()
    missing_test = np.isnan(X_test).sum()
    logger.info(
        f"  After imputation — Train NaN: {missing_train}, Test NaN: {missing_test}")

    # ── 7. Scale features (FIT ON TRAIN ONLY) ────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"  Scaling applied (fit on train only)")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders, feature_cols
