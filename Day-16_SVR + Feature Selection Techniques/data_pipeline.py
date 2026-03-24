"""Day 16: Telomere Length Prediction — Data Pipeline"""
import logging, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_regression, f_regression, RFECV
)
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
import config

logger = logging.getLogger(__name__)


def load_data():
    logger.info("=" * 60)
    logger.info("LOADING TELOMERE LENGTH DATASET")
    logger.info("=" * 60)

    rng = np.random.RandomState(config.RANDOM_SEED)
    n = 3500

    # Biologically meaningful features
    age = rng.normal(50, 15, n).clip(20, 85).astype(np.float32)
    sex = rng.binomial(1, 0.48, n).astype(np.float32)
    bmi = rng.normal(26, 5, n).clip(16, 45).astype(np.float32)
    whr = (0.7 + bmi * 0.005 + rng.normal(0, 0.04, n)).clip(0.6, 1.1).astype(np.float32)
    sys_bp = (110 + age * 0.4 + bmi * 0.3 + rng.normal(0, 12, n)).clip(85, 200).astype(np.float32)
    dia_bp = (sys_bp * 0.6 + rng.normal(0, 6, n)).clip(50, 120).astype(np.float32)
    rhr = (70 - 0.5 * rng.exponential(3, n) + age * 0.1 + rng.normal(0, 8, n)).clip(45, 110).astype(np.float32)
    vo2 = (55 - age * 0.3 - bmi * 0.4 + rng.normal(0, 5, n)).clip(15, 65).astype(np.float32)
    sleep = rng.normal(7, 1.2, n).clip(3, 11).astype(np.float32)
    stress = rng.randint(1, 11, n).astype(np.float32)

    # Lifestyle
    smoking = rng.exponential(5, n).clip(0, 60).astype(np.float32)
    alcohol = rng.exponential(4, n).clip(0, 40).astype(np.float32)
    exercise = rng.exponential(3, n).clip(0, 15).astype(np.float32)

    # Immune/inflammatory markers
    wbc = rng.normal(7, 2, n).clip(2, 15).astype(np.float32)
    lymph = rng.normal(32, 7, n).clip(10, 55).astype(np.float32)
    neut = (100 - lymph - rng.normal(10, 3, n)).clip(30, 80).astype(np.float32)
    crp = rng.lognormal(0.5, 0.8, n).clip(0.1, 30).astype(np.float32)
    il6 = rng.lognormal(0.7, 0.6, n).clip(0.5, 20).astype(np.float32)
    tnf = rng.lognormal(0.5, 0.5, n).clip(0.3, 15).astype(np.float32)
    ox_stress = (crp * 0.3 + il6 * 0.2 + rng.normal(3, 1.5, n)).clip(0, 12).astype(np.float32)

    # Nutritional
    vit_d = rng.normal(30, 12, n).clip(5, 80).astype(np.float32)
    folate = rng.normal(12, 4, n).clip(3, 30).astype(np.float32)
    b12 = rng.normal(450, 150, n).clip(100, 1200).astype(np.float32)
    homocyst = (8 + age * 0.05 + rng.normal(0, 3, n)).clip(3, 30).astype(np.float32)

    # Metabolic
    hdl = (55 - bmi * 0.3 + sex * 10 + rng.normal(0, 10, n)).clip(20, 100).astype(np.float32)
    ldl = (100 + bmi * 1.5 + rng.normal(0, 25, n)).clip(40, 250).astype(np.float32)
    trig = rng.lognormal(4.6, 0.5, n).clip(40, 500).astype(np.float32)
    glucose = (85 + bmi * 0.8 + age * 0.15 + rng.normal(0, 12, n)).clip(60, 250).astype(np.float32)
    hba1c = (glucose * 0.025 + rng.normal(1.5, 0.3, n)).clip(4, 12).astype(np.float32)

    # Hormonal
    cortisol = rng.normal(15, 5, n).clip(3, 40).astype(np.float32)
    dhea = (400 - age * 4 + rng.normal(0, 50, n)).clip(30, 600).astype(np.float32)
    igf1 = (250 - age * 2 + rng.normal(0, 40, n)).clip(50, 500).astype(np.float32)
    telomerase = (0.8 - age * 0.005 + exercise * 0.02 + rng.normal(0, 0.15, n)).clip(0.1, 2.0).astype(np.float32)

    # NOISE features (should be eliminated by feature selection)
    shoe_size = rng.normal(9, 1.5, n).astype(np.float32)
    eye_color = rng.randint(0, 4, n).astype(np.float32)
    birth_month = rng.randint(1, 13, n).astype(np.float32)
    zip_digit = rng.randint(0, 10, n).astype(np.float32)
    fav_number = rng.randint(1, 100, n).astype(np.float32)
    noise1 = rng.normal(0, 1, n).astype(np.float32)
    noise2 = rng.uniform(-1, 1, n).astype(np.float32)

    X = np.column_stack([
        age, sex, bmi, whr, sys_bp, dia_bp, rhr, vo2, sleep, stress,
        smoking, alcohol, exercise, wbc, lymph, neut, crp, il6, tnf, ox_stress,
        vit_d, folate, b12, homocyst, hdl, ldl, trig, glucose, hba1c,
        cortisol, dhea, igf1, telomerase,
        shoe_size, eye_color, birth_month, zip_digit, fav_number, noise1, noise2
    ])

    # Telomere length (kilobases): shorter with age, inflammation, stress
    y = (
        8.5
        - 0.04 * age
        - 0.015 * smoking
        + 0.02 * exercise
        + 0.015 * vo2
        - 0.05 * crp
        - 0.03 * il6
        - 0.02 * ox_stress
        + 0.01 * vit_d
        - 0.02 * homocyst
        - 0.01 * cortisol
        + 0.002 * dhea
        + 0.4 * telomerase
        - 0.01 * stress
        + 0.02 * sleep
        - 0.005 * bmi
        + 0.001 * age * exercise  # interaction: exercise helps more in older age
        + rng.normal(0, 0.35, n)
    ).clip(3, 14).astype(np.float32)

    logger.info(f"Generated {n} subjects | {X.shape[1]} features ({X.shape[1]-7} real + 7 noise)")
    logger.info(f"Telomere length: mean={y.mean():.2f}kb, range=[{y.min():.1f}, {y.max():.1f}]")
    return X, y


def explore_data(X, y):
    logger.info("-" * 60)
    logger.info("EDA")
    logger.info("-" * 60)

    corrs = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    sorted_idx = np.argsort(np.abs(corrs))[::-1]

    # Flag noise features
    n_real = len(config.FEATURE_NAMES) - 7
    logger.info(f"Top correlations with telomere length:")
    for i in sorted_idx[:15]:
        flag = "⚠️ NOISE" if i >= n_real else "✅"
        logger.info(f"  {flag} {config.FEATURE_NAMES[i]:25s} r = {corrs[i]:+.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Target distribution
    axes[0, 0].hist(y, bins=40, color='#7E57C2', edgecolor='white', alpha=0.85)
    axes[0, 0].set_title("🧬 Telomere Length Distribution", fontweight='bold')
    axes[0, 0].set_xlabel("Length (kb)"); axes[0, 0].spines[['top', 'right']].set_visible(False)

    # Top 5 real predictors
    real_top5 = [i for i in sorted_idx if i < n_real][:5]
    for ax_i, fi in enumerate(real_top5):
        ax = axes.ravel()[ax_i + 1]
        ax.scatter(X[:, fi], y, alpha=0.08, s=6, color='#4FC3F7', rasterized=True)
        ax.set_title(f"🔬 {config.FEATURE_NAMES[fi]} (r={corrs[fi]:+.3f})", fontweight='bold', fontsize=10)
        ax.set_ylabel("Telomere (kb)"); ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle("Telomere Length — Target & Top Predictors", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_eda.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_eda.png")


def run_feature_selection_comparison(X_train, y_train, X_test):
    """
    Compare 6 feature selection methods head-to-head.
    The core experiment: which method best separates signal from noise?
    """
    logger.info("=" * 60)
    logger.info("FEATURE SELECTION — 6 Methods Compared")
    logger.info("=" * 60)

    K = config.TOP_K_FEATURES
    results = {}
    n_real = len(config.FEATURE_NAMES) - 7
    noise_indices = set(range(n_real, X_train.shape[1]))

    for method in config.SELECTION_METHODS:
        if method == "none":
            mask = np.ones(X_train.shape[1], dtype=bool)
            selected = set(range(X_train.shape[1]))

        elif method == "variance":
            sel = VarianceThreshold(threshold=0.01)
            sel.fit(X_train)
            mask = sel.get_support()
            # Still might keep too many — take top K by variance
            variances = np.var(X_train, axis=0)
            top_k = np.argsort(variances)[::-1][:K]
            mask = np.zeros(X_train.shape[1], dtype=bool)
            mask[top_k] = True
            selected = set(top_k)

        elif method == "correlation":
            corrs = np.array([abs(np.corrcoef(X_train[:, i], y_train)[0, 1])
                              for i in range(X_train.shape[1])])
            top_k = np.argsort(corrs)[::-1][:K]
            mask = np.zeros(X_train.shape[1], dtype=bool)
            mask[top_k] = True
            selected = set(top_k)

        elif method == "mutual_info":
            mi = mutual_info_regression(X_train, y_train, random_state=config.RANDOM_SEED, n_neighbors=5)
            top_k = np.argsort(mi)[::-1][:K]
            mask = np.zeros(X_train.shape[1], dtype=bool)
            mask[top_k] = True
            selected = set(top_k)

        elif method == "rfecv":
            # Use lightweight SVR for RFE (linear kernel for speed)
            estimator = SVR(kernel="linear", C=1.0)
            rfe = RFECV(estimator, step=3, cv=3, scoring="neg_mean_squared_error",
                         min_features_to_select=K, n_jobs=-1)
            rfe.fit(X_train, y_train)
            mask = rfe.support_
            selected = set(np.where(mask)[0])

        elif method == "lasso":
            lasso = Lasso(alpha=0.01, max_iter=3000, random_state=config.RANDOM_SEED)
            lasso.fit(X_train, y_train)
            importance = np.abs(lasso.coef_)
            top_k = np.argsort(importance)[::-1][:K]
            mask = np.zeros(X_train.shape[1], dtype=bool)
            mask[top_k] = True
            selected = set(top_k)

        # How many noise features survived?
        noise_kept = len(selected & noise_indices)
        real_kept = len(selected - noise_indices)

        results[method] = {
            "mask": mask,
            "selected": selected,
            "n_selected": int(mask.sum()),
            "noise_kept": noise_kept,
            "real_kept": real_kept,
            "names": [config.FEATURE_NAMES[i] for i in sorted(selected) if i < len(config.FEATURE_NAMES)]
        }

        logger.info(f"  {method:15s} → {mask.sum():>3d} features | Real: {real_kept:>2d} | Noise: {noise_kept} {'🔴' if noise_kept > 0 else '✅'}")

    _plot_selection_comparison(results)
    return results


def _plot_selection_comparison(results):
    methods = [m for m in config.SELECTION_METHODS if m != "none"]
    real_counts = [results[m]["real_kept"] for m in methods]
    noise_counts = [results[m]["noise_kept"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (0) Stacked bar: real vs noise features kept
    x = np.arange(len(methods))
    axes[0].bar(x, real_counts, color='#66BB6A', label='Real features ✅', edgecolor='white')
    axes[0].bar(x, noise_counts, bottom=real_counts, color='#EF5350', label='Noise features 🔴', edgecolor='white')
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, fontsize=10)
    axes[0].set_ylabel("# Features Selected")
    axes[0].set_title("🧬 Feature Selection: Signal vs Noise Separation\n(Good methods keep green, reject red)", fontweight='bold')
    axes[0].legend(fontsize=10); axes[0].spines[['top', 'right']].set_visible(False)
    for i, (r, n) in enumerate(zip(real_counts, noise_counts)):
        axes[0].text(i, r + n + 0.3, f'{r}/{r+n}', ha='center', fontweight='bold', fontsize=10)

    # (1) Heatmap: which features each method selected
    feat_names = config.FEATURE_NAMES
    n_feat = len(feat_names)
    heat = np.zeros((len(methods), n_feat))
    for i, m in enumerate(methods):
        for j in results[m]["selected"]:
            if j < n_feat:
                heat[i, j] = 1

    # Sort features by how many methods selected them
    feat_popularity = heat.sum(axis=0)
    sorted_feat = np.argsort(feat_popularity)[::-1][:20]

    im = axes[1].imshow(heat[:, sorted_feat], cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    axes[1].set_yticks(range(len(methods))); axes[1].set_yticklabels(methods, fontsize=10)
    axes[1].set_xticks(range(len(sorted_feat)))
    axes[1].set_xticklabels([feat_names[i][:10] for i in sorted_feat], fontsize=7, rotation=60, ha='right')
    axes[1].set_title("📊 Feature Selection Agreement\n(Bright = selected by method)", fontweight='bold')

    # Mark noise features
    for j_pos, j_feat in enumerate(sorted_feat):
        if j_feat >= n_feat - 7:
            axes[1].axvline(j_pos, color='red', lw=2, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_feature_selection.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_feature_selection.png")


def preprocess_and_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    logger.info(f"Split: Train={X_train.shape[0]} | Test={X_test.shape[0]} | Features={X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, scaler
