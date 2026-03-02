"""
Day 2 - Breast Cancer Diagnosis
Data Pipeline: Loading, EDA, Correlation Analysis, Splitting.

WHAT'S NEW vs Day 1:
---------------------
1. Using sklearn's built-in dataset (no download needed, real clinical data)
2. Correlation analysis — understanding feature redundancy
3. NO SCALING needed for Decision Trees (scale-invariant!)
4. Feature distribution analysis to understand the data geometry

WHY NO SCALING FOR TREES?
---------------------------
Decision Trees split by asking "Is feature X > threshold T?"
This threshold is found by trying all possible values. The SCALE of X
doesn't matter — whether age is in years (0-100) or days (0-36500),
the optimal split point adjusts accordingly.

Compare this to Logistic Regression (Day 1) where the regularization
penalty treats all features equally, so unscaled features with large
magnitudes get unfairly penalized.

RULE OF THUMB:
  - Distance-based models (LR, SVM, KNN) → NEED scaling
  - Tree-based models (DT, RF, XGBoost) → DON'T need scaling
  - Neural networks → NEED scaling (for stable gradient flow)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import logging
import os

from config import DATA_DIR, PLOT_DIR, RANDOM_SEED, TEST_SIZE

logger = logging.getLogger(__name__)


# ============================================================
# 1. DATA LOADING
# ============================================================
def load_data():
    """
    Load the Wisconsin Breast Cancer dataset from sklearn.

    THE DATASET:
    -------------
    This is real clinical data from the University of Wisconsin. Each sample
    represents a fine needle aspirate (FNA) of a breast mass. A digitized
    image of the FNA is analyzed, and 10 properties of cell nuclei are
    computed from the image:

    1. radius       (mean distance from center to perimeter)
    2. texture      (standard deviation of gray-scale values)
    3. perimeter
    4. area
    5. smoothness   (local variation in radius lengths)
    6. compactness  (perimeter² / area - 1.0)
    7. concavity    (severity of concave portions of the contour)
    8. concave points (number of concave portions)
    9. symmetry
    10. fractal dimension ("coastline approximation" - 1)

    For each property, 3 values are recorded:
    - mean: average across all nuclei in the sample
    - se: standard error
    - worst: largest value (mean of 3 largest nuclei)

    Total: 10 properties × 3 statistics = 30 features

    WHY THIS DATASET IS EXCELLENT FOR LEARNING:
    - Real medical data with genuine clinical significance
    - Clean (no missing values)
    - Good class separation (trees can find clear splits)
    - Enough features (30) to learn about feature selection and redundancy
    - Moderate size (569) — big enough for ML, small enough to understand
    """
    data = load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    # sklearn encodes: 0 = malignant, 1 = benign
    # This is REVERSED from medical convention (1 usually = disease)
    # We'll keep sklearn's encoding but be explicit about it

    df['target_name'] = df['target'].map({0: 'malignant', 1: 'benign'})

    logger.info(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1] - 1} columns")
    logger.info(f"Feature names: {list(data.feature_names)}")
    logger.info(f"Target encoding: 0=malignant (cancer), 1=benign (healthy)")

    return df, data.feature_names


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
def explore_data(df):
    """
    Comprehensive EDA with correlation analysis.

    NEW CONCEPT - CORRELATION ANALYSIS:
    -------------------------------------
    With 30 features, many are correlated (e.g., radius, perimeter, and area
    all measure "size"). Correlation analysis reveals:
    1. Which features carry redundant information
    2. Which features are most discriminative for the target
    3. Potential multicollinearity issues (matters for LR, not for trees)

    HOW TO READ A CORRELATION MATRIX:
    - Values range from -1 to +1
    - +1: perfect positive correlation (as one goes up, so does the other)
    - -1: perfect negative correlation (as one goes up, the other goes down)
    - 0: no linear relationship
    - |r| > 0.8: strong correlation — features may be redundant
    """
    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    # Basic info
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Duplicates: {df.duplicated().sum()}")

    # Class distribution
    # NOTE: In this dataset, benign (1) is the majority class
    target_counts = df['target_name'].value_counts()
    target_pct = df['target_name'].value_counts(normalize=True) * 100
    logger.info(f"\nClass distribution:")
    logger.info(f"  Benign (healthy):     {target_counts.get('benign', 0)} ({target_pct.get('benign', 0):.1f}%)")
    logger.info(f"  Malignant (cancer):   {target_counts.get('malignant', 0)} ({target_pct.get('malignant', 0):.1f}%)")

    imbalance_ratio = target_counts.max() / target_counts.min()
    logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 3:
        logger.warning("  Significant imbalance detected! Consider class weights or SMOTE.")
    else:
        logger.info("  Mild imbalance — manageable with stratified splitting.")

    # Feature statistics summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target')
    logger.info(f"\nFeature statistics summary:")
    logger.info(f"  Number of features: {len(numeric_cols)}")
    logger.info(f"  Range of means: {df[numeric_cols].mean().min():.4f} to {df[numeric_cols].mean().max():.2f}")
    logger.info(f"  (This huge range is why scaling matters for LR but NOT for trees)")

    return {
        'shape': df.shape,
        'class_distribution': target_pct.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'n_features': len(numeric_cols)
    }


def plot_correlation_matrix(df, feature_names):
    """
    Visualize feature correlations with a heatmap.

    WHAT TO LOOK FOR:
    ------------------
    1. Bright red/blue blocks = highly correlated feature pairs
    2. If radius_mean, perimeter_mean, and area_mean are all r>0.95,
       they're essentially the same feature measured differently
    3. The bottom row (target) shows which features correlate with the diagnosis

    WHY THIS MATTERS FOR TREES:
    Although trees handle correlated features fine (they just pick one),
    understanding correlations helps you:
    - Interpret feature importance more carefully
    - Know that if the tree picks "worst radius" as important,
      "worst perimeter" and "worst area" are equally informative
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Compute correlation with target for feature ranking
    numeric_df = df.select_dtypes(include=[np.number])
    target_corr = numeric_df.corr()['target'].drop('target').sort_values()

    # Plot: Top correlated features with target
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: Target correlation bar chart
    colors = ['#EF5350' if v < 0 else '#42A5F5' for v in target_corr]
    axes[0].barh(range(len(target_corr)), target_corr.values, color=colors)
    axes[0].set_yticks(range(len(target_corr)))
    axes[0].set_yticklabels(target_corr.index, fontsize=7)
    axes[0].set_xlabel('Correlation with Target', fontsize=11)
    axes[0].set_title('Feature Correlation with Diagnosis\n(Red = ↑Malignant, Blue = ↑Benign)', fontsize=12)
    axes[0].axvline(x=0, color='black', linewidth=0.8)
    axes[0].grid(True, alpha=0.3, axis='x')

    # Right: Compact heatmap of top features
    # Select top 15 most correlated features with target
    top_features = target_corr.abs().nlargest(15).index.tolist() + ['target']
    corr_subset = numeric_df[top_features].corr()

    mask = np.triu(np.ones_like(corr_subset, dtype=bool), k=1)
    sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=axes[1],
                annot_kws={"size": 6})
    axes[1].set_title('Correlation Matrix (Top 15 Features)', fontsize=12)
    axes[1].tick_params(axis='both', labelsize=7)

    plt.suptitle('Day 2: Feature Correlation Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(PLOT_DIR, "correlation_analysis.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Correlation analysis saved to: {filepath}")

    # Log key findings
    logger.info("\nTop 5 features correlated with MALIGNANT diagnosis:")
    for feat, corr_val in target_corr.head(5).items():
        logger.info(f"  {feat:>30}: r = {corr_val:+.4f}")

    logger.info("\nTop 5 features correlated with BENIGN diagnosis:")
    for feat, corr_val in target_corr.tail(5).items():
        logger.info(f"  {feat:>30}: r = {corr_val:+.4f}")

    # Detect highly correlated feature pairs
    corr_matrix = numeric_df.drop('target', axis=1).corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(col, upper_tri.index[i], upper_tri.iloc[i][col])
                       for col in upper_tri.columns
                       for i in range(len(upper_tri))
                       if upper_tri.iloc[i][col] > 0.9]
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

    if high_corr_pairs:
        logger.info(f"\nHighly correlated feature pairs (r > 0.9): {len(high_corr_pairs)} pairs")
        for f1, f2, r in high_corr_pairs[:10]:
            logger.info(f"  {f1:>30} ↔ {f2:<30} r={r:.3f}")
    else:
        logger.info("\nNo highly correlated feature pairs found (r > 0.9)")


# ============================================================
# 3. FEATURE DISTRIBUTION VISUALIZATION
# ============================================================
def plot_feature_distributions(df, feature_names):
    """
    Plot distributions of key features, split by diagnosis.

    WHY THIS MATTERS:
    ------------------
    If a feature has completely overlapping distributions for malignant vs benign,
    it has NO discriminative power. If the distributions are well-separated,
    a Decision Tree can easily split on that feature.

    This visual intuition maps DIRECTLY to how trees work:
    - Well-separated distributions → tree needs only 1 split → low depth
    - Overlapping distributions → tree needs many splits → high depth → overfitting risk
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Select 6 most informative features (3 "mean" + 3 "worst")
    key_features = [
        'mean radius', 'mean concave points', 'mean texture',
        'worst radius', 'worst concave points', 'worst perimeter'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for idx, feat in enumerate(key_features):
        ax = axes[idx]
        for label, color, name in [(0, '#EF5350', 'Malignant'), (1, '#42A5F5', 'Benign')]:
            subset = df[df['target'] == label][feat]
            ax.hist(subset, bins=25, alpha=0.6, color=color, label=name, density=True)
            ax.axvline(subset.mean(), color=color, linestyle='--', linewidth=1.5)

        ax.set_title(feat.title(), fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Feature Distributions by Diagnosis\n(Dashed lines = class means)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(PLOT_DIR, "feature_distributions.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature distributions saved to: {filepath}")


# ============================================================
# 4. PREPROCESSING AND SPLITTING
# ============================================================
def preprocess_and_split(df, feature_names):
    """
    Split data into train/test sets.

    NO SCALING NEEDED FOR DECISION TREES!
    ----------------------------------------
    This is a key difference from Day 1. Since trees split on thresholds,
    the absolute scale of features doesn't matter.

    However, we still MUST:
    1. Split BEFORE any analysis that could leak test info
    2. Use stratification (63% benign / 37% malignant is mild imbalance)
    3. Keep feature names for interpretability

    COMMON MISTAKE WITH TREES:
    Forgetting to set random_state in the DecisionTreeClassifier.
    Trees involve random tie-breaking when two features give equal splits.
    Without a fixed seed, you get different trees on each run!
    """
    X = df[feature_names].copy()
    y = df['target'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    logger.info(f"\nData Split (stratified):")
    logger.info(f"  Train: {X_train.shape[0]} samples")
    logger.info(f"  Test:  {X_test.shape[0]} samples")
    logger.info(f"  Train class balance: benign={y_train.mean():.3f}, malignant={1-y_train.mean():.3f}")
    logger.info(f"  Test class balance:  benign={y_test.mean():.3f}, malignant={1-y_test.mean():.3f}")
    logger.info(f"  No scaling applied (trees are scale-invariant)")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(feature_names)
    }
