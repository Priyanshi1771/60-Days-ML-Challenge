"""Day 19: Radiosensitivity — Data Pipeline (Two Datasets + Cross-Validation)"""
import logging, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import config

logger = logging.getLogger(__name__)


def _generate_dataset(rng, n, shift=0.0, noise_scale=1.0):
    """Generate one radiosensitivity dataset with genomic + clinical features."""
    tp53 = rng.binomial(1, 0.35, n).astype(np.float32)
    genes = rng.normal(0, 1, (n, 9)).astype(np.float32)
    tumor_size = rng.lognormal(1.5, 0.5, n).clip(0.5, 15).astype(np.float32)
    grade = rng.choice([1, 2, 3], n, p=[0.2, 0.45, 0.35]).astype(np.float32)
    stage = rng.choice([1, 2, 3, 4], n, p=[0.15, 0.35, 0.30, 0.20]).astype(np.float32)
    age = rng.normal(58 + shift, 12, n).clip(20, 85).astype(np.float32)
    ki67 = rng.normal(30, 15, n).clip(1, 90).astype(np.float32)
    scores = rng.normal(0, 1, (n, 15)).astype(np.float32)

    X = np.column_stack([tp53, genes, tumor_size, grade, stage, age, ki67, scores])

    # Target: Surviving Fraction at 2Gy (SF2)
    y = (
        0.45 - 0.08 * tp53 + 0.05 * genes[:, 0] - 0.03 * genes[:, 3]
        - 0.02 * tumor_size - 0.05 * grade + 0.01 * age * 0.01
        - 0.04 * scores[:, 0] + 0.03 * scores[:, 5] + shift * 0.05
        + rng.normal(0, 0.08 * noise_scale, n)
    ).clip(0.05, 0.95).astype(np.float32)
    return X, y


def load_datasets():
    logger.info("=" * 60)
    logger.info("LOADING TWO RADIOSENSITIVITY DATASETS")
    logger.info("=" * 60)

    rng = np.random.RandomState(config.RANDOM_SEED)
    X_a, y_a = _generate_dataset(rng, config.DATASET_A_SAMPLES, shift=0.0, noise_scale=1.0)
    X_b, y_b = _generate_dataset(rng, config.DATASET_B_SAMPLES, shift=2.0, noise_scale=1.3)

    logger.info(f"Dataset A (cell lines): {X_a.shape[0]} | SF2 mean={y_a.mean():.3f}")
    logger.info(f"Dataset B (clinical):   {X_b.shape[0]} | SF2 mean={y_b.mean():.3f}")
    logger.info(f"Domain shift: {abs(y_a.mean() - y_b.mean()):.3f}")
    return X_a, y_a, X_b, y_b


def explore_data(X_a, y_a, X_b, y_b):
    logger.info("-" * 60)
    logger.info("EDA")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(y_a, bins=30, alpha=0.7, color='#4FC3F7', label='A (cell lines)', edgecolor='white')
    axes[0].hist(y_b, bins=30, alpha=0.7, color='#FF7043', label='B (clinical)', edgecolor='white')
    axes[0].set_title("SF2 Distributions (Domain Shift!)", fontweight='bold')
    axes[0].set_xlabel("Surviving Fraction"); axes[0].legend()
    axes[0].spines[['top', 'right']].set_visible(False)

    corrs_a = [np.corrcoef(X_a[:, i], y_a)[0, 1] for i in range(X_a.shape[1])]
    corrs_b = [np.corrcoef(X_b[:, i], y_b)[0, 1] for i in range(X_b.shape[1])]
    axes[1].scatter(corrs_a, corrs_b, alpha=0.6, s=30, color='#7E57C2')
    axes[1].plot([-0.5, 0.5], [-0.5, 0.5], 'r--', lw=1)
    axes[1].set_xlabel("Corr in A"); axes[1].set_ylabel("Corr in B")
    axes[1].set_title("Feature Correlations: A vs B", fontweight='bold')
    axes[1].spines[['top', 'right']].set_visible(False)

    top8 = np.argsort(np.abs(corrs_a))[::-1][:8]
    x_pos = np.arange(8)
    axes[2].bar(x_pos - 0.2, [corrs_a[i] for i in top8], 0.35, color='#4FC3F7', label='A')
    axes[2].bar(x_pos + 0.2, [corrs_b[i] for i in top8], 0.35, color='#FF7043', label='B')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([config.FEATURE_NAMES[i][:10] for i in top8], fontsize=7, rotation=30)
    axes[2].set_title("Top Features by Dataset", fontweight='bold')
    axes[2].legend(fontsize=8); axes[2].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_eda.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_eda.png")


def prepare_splits(X_a, y_a, X_b, y_b):
    """Three validation strategies: within-A, within-B, cross-dataset A->B."""
    logger.info("-" * 60)
    logger.info("PREPARING 3 VALIDATION STRATEGIES")

    scaler_a = StandardScaler()
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        X_a, y_a, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    Xa_tr = scaler_a.fit_transform(Xa_tr).astype(np.float32)
    Xa_te = scaler_a.transform(Xa_te).astype(np.float32)

    # Cross-dataset: train A, test B (scaled with A's scaler)
    Xb_cross = scaler_a.transform(X_b).astype(np.float32)

    scaler_b = StandardScaler()
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        X_b, y_b, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    Xb_tr = scaler_b.fit_transform(Xb_tr).astype(np.float32)
    Xb_te = scaler_b.transform(Xb_te).astype(np.float32)

    logger.info(f"  within_A: Train {Xa_tr.shape[0]} -> Test {Xa_te.shape[0]}")
    logger.info(f"  within_B: Train {Xb_tr.shape[0]} -> Test {Xb_te.shape[0]}")
    logger.info(f"  cross_AB: Train {Xa_tr.shape[0]} -> Test {X_b.shape[0]} [CROSS-DATASET]")

    return {
        "within_A": (Xa_tr, Xa_te, ya_tr, ya_te),
        "within_B": (Xb_tr, Xb_te, yb_tr, yb_te),
        "cross_A_to_B": (Xa_tr, Xb_cross, ya_tr, y_b),
    }, scaler_a
