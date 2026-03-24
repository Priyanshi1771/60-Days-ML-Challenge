"""Day 18: Gene Expression Prediction — Data Pipeline"""
import logging, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import config

logger = logging.getLogger(__name__)


def load_data():
    logger.info("=" * 60)
    logger.info("LOADING GENE EXPRESSION DATASET")
    logger.info("=" * 60)

    rng = np.random.RandomState(config.RANDOM_SEED)
    n = config.N_SAMPLES
    n_total = config.N_GENES_INPUT
    n_signal = n_total - config.NOISE_GENES

    # Signal genes: correlated with target through latent biological pathways
    # Simulate 5 latent pathways that drive gene expression
    n_pathways = 5
    pathway_strengths = rng.uniform(0.3, 1.0, n_pathways).astype(np.float32)

    latent = rng.normal(0, 1, (n, n_pathways)).astype(np.float32)

    # Signal genes are linear combinations of latent pathways + noise
    signal_loadings = rng.normal(0, 0.5, (n_pathways, n_signal)).astype(np.float32)
    X_signal = (latent @ signal_loadings + rng.normal(0, 0.3, (n, n_signal))).astype(np.float32)

    # Noise genes: completely random, no relationship to target
    X_noise = rng.normal(0, 1, (n, config.NOISE_GENES)).astype(np.float32)

    X = np.hstack([X_signal, X_noise])

    # Target: nonlinear function of latent pathways
    y = (
        2.0 * latent[:, 0]
        + 1.5 * np.tanh(latent[:, 1] * 2)
        - 1.0 * latent[:, 2] ** 2
        + 0.8 * latent[:, 3] * latent[:, 4]  # interaction
        + 0.5 * np.sin(latent[:, 0] * np.pi)  # nonlinearity
        + rng.normal(0, 0.5, n)
    ).astype(np.float32)

    # Shuffle column order so signal/noise are mixed
    col_order = rng.permutation(n_total)
    X = X[:, col_order]

    # Track which columns are signal vs noise (for evaluation)
    is_signal = np.zeros(n_total, dtype=bool)
    is_signal[:n_signal] = True
    is_signal = is_signal[col_order]

    gene_names = [f"Gene_{i:04d}" for i in range(n_total)]

    logger.info(f"Generated {n} samples | {n_total} genes ({n_signal} signal + {config.NOISE_GENES} noise)")
    logger.info(f"Target: mean={y.mean():.3f}, std={y.std():.3f}, range=[{y.min():.2f}, {y.max():.2f}]")
    logger.info(f"5 latent biological pathways drive expression")
    return X, y, is_signal, gene_names


def explore_data(X, y, is_signal):
    logger.info("-" * 60)
    logger.info("EDA")
    logger.info("-" * 60)

    # Correlations with target
    corrs = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    signal_corrs = np.abs(corrs[is_signal])
    noise_corrs = np.abs(corrs[~is_signal])

    logger.info(f"Signal genes: mean |r| = {signal_corrs.mean():.4f}, max = {signal_corrs.max():.4f}")
    logger.info(f"Noise genes:  mean |r| = {noise_corrs.mean():.4f}, max = {noise_corrs.max():.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Target distribution
    axes[0, 0].hist(y, bins=40, color='#7E57C2', edgecolor='white', alpha=0.85)
    axes[0, 0].set_title("DNA Target Gene Expression", fontweight='bold')
    axes[0, 0].set_xlabel("Expression Level"); axes[0, 0].spines[['top', 'right']].set_visible(False)

    # Correlation distribution: signal vs noise
    axes[0, 1].hist(signal_corrs, bins=30, alpha=0.7, color='#66BB6A', label=f'Signal ({is_signal.sum()})', edgecolor='white')
    axes[0, 1].hist(noise_corrs, bins=30, alpha=0.7, color='#EF5350', label=f'Noise ({(~is_signal).sum()})', edgecolor='white')
    axes[0, 1].set_title("|Correlation| with Target: Signal vs Noise", fontweight='bold')
    axes[0, 1].set_xlabel("|r|"); axes[0, 1].legend(); axes[0, 1].spines[['top', 'right']].set_visible(False)

    # PCA: 2D projection colored by target
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    X_pca = pca.fit_transform(X)
    sc = axes[0, 2].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.3, s=8, rasterized=True)
    axes[0, 2].set_title(f"PCA Projection (var={pca.explained_variance_ratio_.sum()*100:.1f}%)", fontweight='bold')
    axes[0, 2].set_xlabel("PC1"); axes[0, 2].set_ylabel("PC2")
    plt.colorbar(sc, ax=axes[0, 2], label='Expression')
    axes[0, 2].spines[['top', 'right']].set_visible(False)

    # Top 5 correlated genes scatter
    top5 = np.argsort(np.abs(corrs))[::-1][:5]
    for ax_i, gi in enumerate(top5[:3]):
        ax = axes[1, ax_i]
        color = '#66BB6A' if is_signal[gi] else '#EF5350'
        tag = "SIGNAL" if is_signal[gi] else "NOISE"
        ax.scatter(X[:, gi], y, alpha=0.1, s=6, color=color, rasterized=True)
        ax.set_title(f"Gene_{gi:04d} (r={corrs[gi]:.3f}) [{tag}]", fontweight='bold', fontsize=10)
        ax.set_ylabel("Expression"); ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle("Gene Expression Data -- High-Dimensional Genomics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_eda.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_eda.png")


def preprocess_and_split(X, y):
    """Three-way split: train/val/test. Scale fit on train only."""
    logger.info("-" * 60)
    logger.info("PREPROCESSING")
    logger.info("-" * 60)

    # First split off test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)

    # Then split train/val from remainder
    val_frac = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=config.RANDOM_SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    logger.info(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def to_tensors(X, y, device=config.DEVICE):
    return torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
