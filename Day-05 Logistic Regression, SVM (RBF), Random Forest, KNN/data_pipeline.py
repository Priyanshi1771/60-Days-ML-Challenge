"""
=============================================================================
 Day 5: Thyroid Disease Classification — Data Pipeline
=============================================================================
 Handles: Loading UCI New Thyroid dataset, EDA, preprocessing, train/test split
 
 KEY PRINCIPLE: Fit scalers on train set ONLY — never on full data.
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml

import config

logger = logging.getLogger(__name__)


def load_data():
    """
    Load the UCI New Thyroid dataset from OpenML.
    
    The dataset contains 215 instances with 5 features measuring thyroid
    function, classified into 3 classes:
        1 = Normal (150 samples)
        2 = Hyperthyroid (35 samples)  
        3 = Hypothyroid (30 samples)
    
    This is an IMBALANCED dataset — important for evaluation strategy.
    """
    logger.info("=" * 60)
    logger.info("LOADING UCI NEW THYROID DATASET")
    logger.info("=" * 60)
    
    try:
        # Fetch from OpenML (dataset ID 40 = new-thyroid)
        thyroid = fetch_openml(name="new-thyroid", version=1, as_frame=True, parser="auto")
        df = thyroid.data.copy()
        df[config.TARGET_NAME] = thyroid.target
        
        logger.info(f"Dataset loaded successfully from OpenML")
    except Exception as e:
        logger.warning(f"OpenML fetch failed: {e}")
        logger.info("Generating synthetic thyroid-like dataset as fallback...")
        df = _generate_fallback_data()
    
    # Ensure proper column names
    if list(df.columns[:-1]) != config.FEATURE_NAMES:
        feature_cols = list(df.columns[:-1])
        rename_map = {old: new for old, new in zip(feature_cols, config.FEATURE_NAMES)}
        df.rename(columns=rename_map, inplace=True)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features: {config.FEATURE_NAMES}")
    logger.info(f"Target: {config.TARGET_NAME}")
    
    return df


def _generate_fallback_data():
    """
    Generate synthetic data mimicking UCI New Thyroid distribution.
    Used only if OpenML is unreachable.
    """
    np.random.seed(config.RANDOM_SEED)
    
    # Normal thyroid (class 1) - 150 samples
    normal = pd.DataFrame({
        "T3_resin_uptake": np.random.normal(110, 12, 150),
        "Total_serum_thyroxin": np.random.normal(10.5, 2.5, 150),
        "Total_serum_triiodothyronine": np.random.normal(1.8, 0.5, 150),
        "TSH": np.random.normal(1.5, 0.8, 150),
        "Max_diff_TSH_after_TRH": np.random.normal(4.0, 2.5, 150),
        "thyroid_class": "1"
    })
    
    # Hyperthyroid (class 2) - 35 samples
    hyper = pd.DataFrame({
        "T3_resin_uptake": np.random.normal(130, 10, 35),
        "Total_serum_thyroxin": np.random.normal(17.0, 3.0, 35),
        "Total_serum_triiodothyronine": np.random.normal(3.5, 1.0, 35),
        "TSH": np.random.normal(0.3, 0.2, 35),
        "Max_diff_TSH_after_TRH": np.random.normal(1.0, 0.8, 35),
        "thyroid_class": "2"
    })
    
    # Hypothyroid (class 3) - 30 samples
    hypo = pd.DataFrame({
        "T3_resin_uptake": np.random.normal(95, 8, 30),
        "Total_serum_thyroxin": np.random.normal(4.5, 1.5, 30),
        "Total_serum_triiodothyronine": np.random.normal(0.8, 0.3, 30),
        "TSH": np.random.normal(15.0, 5.0, 30),
        "Max_diff_TSH_after_TRH": np.random.normal(25.0, 8.0, 30),
        "thyroid_class": "3"
    })
    
    df = pd.concat([normal, hyper, hypo], ignore_index=True)
    return df


def explore_data(df):
    """
    Perform exploratory data analysis and save visualizations.
    
    Generates:
        1. Class distribution bar chart
        2. Feature distributions by class (boxplots)
        3. Correlation heatmap
        4. Pairplot colored by class
    """
    logger.info("-" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("-" * 60)
    
    # Basic stats
    logger.info(f"\nDataset Info:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Features: {len(config.FEATURE_NAMES)}")
    logger.info(f"  Missing values: {df.isnull().sum().sum()}")
    
    # Class distribution
    class_dist = df[config.TARGET_NAME].value_counts().sort_index()
    logger.info(f"\nClass Distribution:")
    for cls, count in class_dist.items():
        pct = count / len(df) * 100
        logger.info(f"  Class {cls}: {count} samples ({pct:.1f}%)")
    
    # Feature statistics
    logger.info(f"\nFeature Statistics:")
    stats = df[config.FEATURE_NAMES].describe()
    logger.info(f"\n{stats.to_string()}")
    
    # ─── Plot 1: Class Distribution ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4FC3F7", "#FF7043", "#66BB6A"]
    bars = ax.bar(config.CLASS_NAMES, class_dist.values, color=colors, edgecolor="white", linewidth=1.5)
    
    for bar, count in zip(bars, class_dist.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{count}\n({count/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Thyroid Disease — Class Distribution", fontsize=14, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Thyroid Class", fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_class_distribution.png")
    
    # ─── Plot 2: Feature Boxplots by Class ───────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for i, feature in enumerate(config.FEATURE_NAMES):
        df.boxplot(column=feature, by=config.TARGET_NAME, ax=axes[i],
                   patch_artist=True,
                   boxprops=dict(facecolor='#4FC3F7', alpha=0.7),
                   medianprops=dict(color='#FF7043', linewidth=2))
        axes[i].set_title(feature.replace('_', ' '), fontsize=10, fontweight='bold')
        axes[i].set_xlabel("Class")
    
    fig.suptitle("Feature Distributions by Thyroid Class", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_feature_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_feature_boxplots.png")
    
    # ─── Plot 3: Correlation Heatmap ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[config.FEATURE_NAMES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True, linewidths=1,
                xticklabels=[f.replace('_', '\n') for f in config.FEATURE_NAMES],
                yticklabels=[f.replace('_', '\n') for f in config.FEATURE_NAMES])
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_correlation_heatmap.png")
    
    return class_dist


def preprocess_and_split(df):
    """
    Preprocess data and create train/test splits.
    
    CRITICAL: StandardScaler is fit ONLY on training data, then
    applied to both train and test. This prevents data leakage.
    
    Returns:
        X_train, X_test: Scaled feature arrays
        y_train, y_test: Encoded target arrays
        scaler: Fitted StandardScaler (for saving/inference)
        label_encoder: Fitted LabelEncoder
    """
    logger.info("-" * 60)
    logger.info("PREPROCESSING & SPLITTING")
    logger.info("-" * 60)
    
    # Separate features and target
    X = df[config.FEATURE_NAMES].values.astype(np.float64)
    y = df[config.TARGET_NAME].values
    
    # Encode target labels (string → int)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logger.info(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Handle missing values (if any)
    if np.isnan(X).any():
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        logger.info("Imputed missing values with column medians")
    
    # ── STRATIFIED Split ─────────────────────────────────────────────────
    # Stratified ensures class proportions are preserved in both splits
    # This is CRITICAL for imbalanced datasets like thyroid
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y_encoded  # <-- preserves class balance
    )
    
    logger.info(f"\nSplit sizes:")
    logger.info(f"  Training: {X_train.shape[0]} samples")
    logger.info(f"  Testing:  {X_test.shape[0]} samples")
    
    # Verify stratification
    for i, cls_name in enumerate(config.CLASS_NAMES):
        train_pct = (y_train == i).sum() / len(y_train) * 100
        test_pct = (y_test == i).sum() / len(y_test) * 100
        logger.info(f"  {cls_name}: Train={train_pct:.1f}% | Test={test_pct:.1f}%")
    
    # ── Scale features ───────────────────────────────────────────────────
    # FIT on train ONLY, TRANSFORM both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
    X_test_scaled = scaler.transform(X_test)          # transform only (NO fit!)
    
    logger.info(f"\nScaling applied (fit on train only):")
    logger.info(f"  Train mean ≈ {X_train_scaled.mean(axis=0).round(4)}")
    logger.info(f"  Train std  ≈ {X_train_scaled.std(axis=0).round(4)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder
