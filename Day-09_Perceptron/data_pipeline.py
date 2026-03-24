"""
=============================================================================
 Day 9: Hepatitis Diagnosis — Data Pipeline (Optimized)
=============================================================================
 CHALLENGE: Only 155 samples with ~48% missing values + class imbalance
 OPTIMIZATION: Vectorized ops, no redundant copies, efficient imputation
=============================================================================
"""
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import config

logger = logging.getLogger(__name__)


def load_data():
    """Load UCI Hepatitis or generate realistic fallback."""
    logger.info("=" * 60)
    logger.info("LOADING UCI HEPATITIS DATASET")
    logger.info("=" * 60)
    try:
        from sklearn.datasets import fetch_openml
        h = fetch_openml(data_id=55, as_frame=True, parser="auto")
        X, y = h.data.values.astype(np.float32), (h.target.astype(int).values)
        # Remap: original 1=Die,2=Live → 0=Die,1=Live
        y = np.where(y == 2, 1, 0).astype(np.int8)
        logger.info("Loaded from OpenML (id=55)")
    except Exception as e:
        logger.warning(f"OpenML failed: {e}")
        X, y = _generate_fallback()
    logger.info(f"Shape: {X.shape} | Target dist: Die={np.sum(y==0)}, Live={np.sum(y==1)}")
    return X, y


def _generate_fallback():
    """Memory-efficient synthetic hepatitis data (float32 + int8)."""
    rng = np.random.RandomState(config.RANDOM_SEED)
    n = 155
    n_die, n_live = 32, 123  # Real class split: ~20% die, ~80% live

    def _col(die_params, live_params, dtype='f'):
        """Generate a single feature column efficiently."""
        if dtype == 'b':  # binary
            d = rng.binomial(1, die_params, n_die).astype(np.float32)
            l = rng.binomial(1, live_params, n_live).astype(np.float32)
        else:
            d = rng.normal(*die_params, n_die).astype(np.float32)
            l = rng.normal(*live_params, n_live).astype(np.float32)
        return np.concatenate([d, l])

    X = np.column_stack([
        _col((46, 12), (38, 13)),                  # Age
        _col(0.85, 0.80, 'b'),                     # Sex (male=1)
        _col(0.45, 0.55, 'b'),                     # Steroid
        _col(0.25, 0.15, 'b'),                     # Antivirals
        _col(0.75, 0.55, 'b'),                     # Fatigue
        _col(0.65, 0.35, 'b'),                     # Malaise
        _col(0.55, 0.25, 'b'),                     # Anorexia
        _col(0.50, 0.60, 'b'),                     # Liver_Big
        _col(0.65, 0.40, 'b'),                     # Liver_Firm
        _col(0.55, 0.20, 'b'),                     # Spleen_Palpable
        _col(0.60, 0.22, 'b'),                     # Spiders
        _col(0.50, 0.12, 'b'),                     # Ascites
        _col(0.35, 0.08, 'b'),                     # Varices
        _col((2.8, 2.0), (1.2, 0.8)).clip(0.3, 8), # Bilirubin
        _col((120, 45), (85, 30)).clip(25, 300),    # Alk_Phosphate
        _col((120, 70), (60, 35)).clip(10, 500),    # SGOT
        _col((3.2, 0.7), (3.9, 0.5)).clip(1.5, 5.5), # Albumin
        _col((45, 20), (65, 15)).clip(10, 100),     # Protime
        _col(0.65, 0.45, 'b'),                     # Histology
    ]).astype(np.float32)

    y = np.concatenate([np.zeros(n_die), np.ones(n_live)]).astype(np.int8)

    # Inject ~5% missing (realistic for clinical data)
    mask = rng.random(X.shape) < 0.05
    X[mask] = np.nan

    # Shuffle together
    idx = rng.permutation(n)
    return X[idx], y[idx]


def explore_data(X, y):
    """Lightweight EDA — 2 plots, minimal memory."""
    logger.info("-" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("-" * 60)

    n_miss = np.isnan(X).sum()
    logger.info(f"Total missing: {n_miss} ({n_miss/X.size*100:.1f}%)")
    logger.info(f"Class dist: Die={np.sum(y==0)} ({np.mean(y==0)*100:.1f}%) | "
                f"Live={np.sum(y==1)} ({np.mean(y==1)*100:.1f}%)")
    logger.info(f"⚠️  Imbalance: {np.sum(y==1)/np.sum(y==0):.1f}:1 (Live:Die)")

    # Per-feature missing
    miss_per_feat = np.isnan(X).sum(axis=0)
    for i, name in enumerate(config.FEATURE_NAMES):
        if miss_per_feat[i] > 0:
            logger.info(f"  {name:20s}: {miss_per_feat[i]:3d} missing ({miss_per_feat[i]/X.shape[0]*100:.1f}%)")

    # ─── Plot 1: Class distribution + key features ──────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Class dist
    counts = [np.sum(y==0), np.sum(y==1)]
    axes[0,0].bar(config.CLASS_NAMES, counts, color=["#EF5350", "#66BB6A"], edgecolor="white")
    for i, cnt in enumerate(counts):
        axes[0,0].text(i, cnt+1, f'{cnt}\n({cnt/len(y)*100:.1f}%)', ha='center', fontweight='bold')
    axes[0,0].set_title("🦠 Class Distribution", fontweight='bold')
    axes[0,0].spines[['top','right']].set_visible(False)

    # Key numeric features (indices: 0=Age, 13=Bilirubin, 15=SGOT, 16=Albumin, 17=Protime)
    feat_idx = [(0, "Age"), (13, "Bilirubin"), (15, "SGOT"), (16, "Albumin"), (17, "Protime")]
    for ax_idx, (fi, fname) in enumerate(feat_idx):
        ax = axes.ravel()[ax_idx + 1]
        for cls, color, label in [(0, "#EF5350", "Die"), (1, "#66BB6A", "Live")]:
            vals = X[y == cls, fi]
            vals = vals[~np.isnan(vals)]
            ax.hist(vals, bins=15, alpha=0.6, color=color, label=label, edgecolor='white')
        ax.set_title(f"🔬 {fname}", fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)

    plt.suptitle("Hepatitis — Key Feature Distributions by Outcome", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_eda_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_eda_overview.png")

    # ─── Plot 2: Correlation of numeric features with outcome ────────
    num_idx = [0, 13, 14, 15, 16, 17]  # numeric feature indices
    num_names = [config.FEATURE_NAMES[i] for i in num_idx] + [config.TARGET_NAME]
    data_for_corr = np.column_stack([X[:, num_idx], y.reshape(-1,1)])
    # Remove NaN rows for correlation
    valid = ~np.isnan(data_for_corr).any(axis=1)
    corr = np.corrcoef(data_for_corr[valid].T)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(num_names)))
    ax.set_xticklabels(num_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(num_names)))
    ax.set_yticklabels(num_names, fontsize=9)
    for i in range(len(num_names)):
        for j in range(len(num_names)):
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(corr[i,j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax)
    ax.set_title("🧬 Correlation: Numeric Features vs Survival", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_correlation.png")


def preprocess_and_split(X, y):
    """
    Preprocess with minimal memory: float32, in-place where possible.
    Impute → Split → Scale (all fit on train only).
    
    NOTE: We split AFTER imputation here because the dataset is tiny (155 samples).
    With only 31 test samples, train-only imputation would be unreliable.
    We use median imputation which is stable enough for this dataset size.
    For larger datasets, always split first then impute.
    """
    logger.info("-" * 60)
    logger.info("PREPROCESSING & SPLITTING")
    logger.info("-" * 60)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y)
    logger.info(f"Split: Train={len(y_train)} | Test={len(y_test)}")
    logger.info(f"  Train die rate: {np.mean(y_train==0):.3f} | Test die rate: {np.mean(y_test==0):.3f}")

    # Impute (median — robust to outliers, fit on train)
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    logger.info(f"  Imputed + scaled (float32) | Train NaN: {np.isnan(X_train).sum()}")
    return X_train, X_test, y_train, y_test, scaler, imputer
