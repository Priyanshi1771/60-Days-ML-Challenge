"""Day 12: Blood Pressure Prediction — Data Pipeline"""
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
    logger.info("=" * 60)
    logger.info("LOADING BLOOD PRESSURE DATASET")
    logger.info("=" * 60)

    rng = np.random.RandomState(config.RANDOM_SEED)
    n = 5000

    # Generate features WITH deliberate multicollinearity
    age = rng.normal(52, 15, n).clip(18, 85).astype(np.float32)
    weight = (rng.normal(78, 16, n) + age * 0.1).clip(40, 150).astype(np.float32)
    height = rng.normal(170, 10, n).clip(140, 200).astype(np.float32)
    bmi = (weight / (height / 100) ** 2).astype(np.float32)  # DERIVED from weight+height → collinear!
    waist = (bmi * 2.5 + rng.normal(0, 5, n)).clip(55, 150).astype(np.float32)  # correlated with BMI
    hr = (rng.normal(72, 12, n) + age * 0.05).clip(45, 120).astype(np.float32)

    chol_total = (rng.normal(200, 38, n) + age * 0.5).clip(100, 350).astype(np.float32)
    chol_ldl = (chol_total * 0.6 + rng.normal(0, 15, n)).clip(40, 250).astype(np.float32)  # subset → collinear!
    chol_hdl = (rng.normal(55, 15, n) - bmi * 0.3).clip(20, 100).astype(np.float32)
    triglycerides = (rng.normal(140, 55, n) + bmi * 2).clip(40, 500).astype(np.float32)

    glucose = (rng.normal(95, 22, n) + bmi * 0.5).clip(60, 300).astype(np.float32)
    creatinine = rng.normal(1.0, 0.3, n).clip(0.4, 3.0).astype(np.float32)
    sodium = (rng.normal(3200, 800, n)).clip(1000, 6000).astype(np.float32)
    potassium = (5000 - sodium * 0.4 + rng.normal(0, 400, n)).clip(1500, 5000).astype(np.float32)  # inversely correlated!

    alcohol = rng.exponential(3, n).clip(0, 30).astype(np.float32)
    exercise = rng.exponential(3, n).clip(0, 20).astype(np.float32)
    sleep = rng.normal(7, 1.2, n).clip(3, 12).astype(np.float32)
    stress = rng.choice(np.arange(1, 11), n).astype(np.float32)
    smoking = rng.binomial(1, 0.22, n).astype(np.float32)
    diabetes = rng.binomial(1, 0.12, n).astype(np.float32)
    family_htn = rng.binomial(1, 0.35, n).astype(np.float32)

    X = np.column_stack([age, weight, height, bmi, waist, hr, chol_total, chol_ldl,
                          chol_hdl, triglycerides, glucose, creatinine, sodium, potassium,
                          alcohol, exercise, sleep, stress, smoking, diabetes, family_htn])

    # Target: systolic BP (nonlinear with interactions)
    systolic = (
        90 + 0.35 * age + 0.12 * bmi + 0.08 * weight * 0.1 + 0.005 * sodium * 0.01
        - 0.15 * exercise + 0.1 * stress + 3.5 * smoking + 4.0 * diabetes
        + 5.0 * family_htn + 0.003 * chol_total - 0.08 * chol_hdl
        + 0.04 * hr + 2.5 * alcohol * 0.1 - 0.3 * sleep
        + 0.001 * age * bmi  # interaction
        + rng.normal(0, 8, n)  # noise
    ).clip(80, 200).astype(np.float32)

    # Diastolic is correlated with systolic
    diastolic = (systolic * 0.55 + rng.normal(10, 5, n)).clip(50, 120).astype(np.float32)

    logger.info(f"Generated {n} patients | {X.shape[1]} features")
    logger.info(f"Systolic: {systolic.mean():.1f} ± {systolic.std():.1f} mmHg")
    logger.info(f"Diastolic: {diastolic.mean():.1f} ± {diastolic.std():.1f} mmHg")
    return X, systolic, diastolic


def explore_data(X, y_sys, y_dia):
    logger.info("-" * 60)
    logger.info("EDA + MULTICOLLINEARITY ANALYSIS")
    logger.info("-" * 60)

    # ─── Plot 1: Target distributions + correlation heatmap ──────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(y_sys, bins=40, color='#EF5350', alpha=0.8, edgecolor='white')
    axes[0].set_title("❤️ Systolic BP Distribution", fontweight='bold')
    axes[0].set_xlabel("mmHg"); axes[0].spines[['top','right']].set_visible(False)

    axes[1].hist(y_dia, bins=40, color='#42A5F5', alpha=0.8, edgecolor='white')
    axes[1].set_title("💙 Diastolic BP Distribution", fontweight='bold')
    axes[1].set_xlabel("mmHg"); axes[1].spines[['top','right']].set_visible(False)

    axes[2].scatter(y_sys, y_dia, alpha=0.15, s=8, color='#AB47BC', rasterized=True)
    r = np.corrcoef(y_sys, y_dia)[0, 1]
    axes[2].set_title(f"💜 Sys vs Dia (r={r:.3f})", fontweight='bold')
    axes[2].set_xlabel("Systolic"); axes[2].set_ylabel("Diastolic")
    axes[2].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_targets.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_targets.png")

    # ─── Plot 2: Multicollinearity heatmap (the KEY visualization) ────
    corr = np.corrcoef(X.T)
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=config.FEATURE_NAMES, yticklabels=config.FEATURE_NAMES,
                ax=ax, square=True, linewidths=0.5, annot_kws={"size": 7},
                vmin=-1, vmax=1)
    ax.set_title("🔍 Feature Correlation Matrix — Multicollinearity Exposed!\n"
                 "(Red clusters = collinear features that break OLS regression)",
                 fontweight='bold', fontsize=13)

    # Highlight collinear pairs
    for f1, f2 in config.COLLINEAR_PAIRS:
        i, j = config.FEATURE_NAMES.index(f1), config.FEATURE_NAMES.index(f2)
        ax.add_patch(plt.Rectangle((min(i,j), min(i,j)), abs(i-j)+1, abs(i-j)+1,
                                    fill=False, edgecolor='lime', lw=3))

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_multicollinearity.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_multicollinearity.png")

    # Log the dangerous correlations
    logger.info("\n⚠️  HIGH CORRELATIONS (|r| > 0.7):")
    for i in range(len(config.FEATURE_NAMES)):
        for j in range(i + 1, len(config.FEATURE_NAMES)):
            r_val = corr[i, j]
            if abs(r_val) > 0.7:
                logger.info(f"  {config.FEATURE_NAMES[i]:25s} ↔ {config.FEATURE_NAMES[j]:25s} | r = {r_val:+.3f} 🔴")


def compute_vif(X):
    """Variance Inflation Factor — THE diagnostic for multicollinearity."""
    logger.info("\n📊 VARIANCE INFLATION FACTOR (VIF):")
    logger.info("   VIF > 5 = moderate | VIF > 10 = severe multicollinearity\n")

    from numpy.linalg import lstsq
    n_features = X.shape[1]
    vifs = np.empty(n_features, dtype=np.float32)

    for i in range(n_features):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        # Add intercept
        X_aug = np.column_stack([np.ones(len(y_i)), X_others])
        coef, _, _, _ = lstsq(X_aug, y_i, rcond=None)
        y_pred = X_aug @ coef
        ss_res = np.sum((y_i - y_pred) ** 2)
        ss_tot = np.sum((y_i - y_i.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vifs[i] = 1 / (1 - r2) if r2 < 1 else 999

    # Sort by VIF descending
    sorted_idx = np.argsort(vifs)[::-1]
    for idx in sorted_idx:
        flag = "🔴 SEVERE" if vifs[idx] > 10 else "🟡 MODERATE" if vifs[idx] > 5 else "🟢 OK"
        logger.info(f"  {config.FEATURE_NAMES[idx]:25s} VIF = {vifs[idx]:8.2f}  {flag}")

    # Plot VIF
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#EF5350' if v > 10 else '#FFB74D' if v > 5 else '#66BB6A' for v in vifs[sorted_idx]]
    ax.barh(range(len(sorted_idx)), vifs[sorted_idx], color=colors, edgecolor='white', alpha=0.85)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([config.FEATURE_NAMES[i] for i in sorted_idx], fontsize=9)
    ax.axvline(x=5, color='#FFB74D', linestyle='--', lw=2, label='VIF=5 (moderate)')
    ax.axvline(x=10, color='#EF5350', linestyle='--', lw=2, label='VIF=10 (severe)')
    ax.set_xlabel("Variance Inflation Factor")
    ax.set_title("📊 VIF Analysis — Which Features Cause Multicollinearity?\n"
                 "(Red > 10 = remove or regularize | Green < 5 = safe)",
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_vif.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_vif.png")

    return vifs


def preprocess_and_split(X, y_sys, y_dia):
    logger.info("-" * 60)
    logger.info("PREPROCESSING")
    logger.info("-" * 60)

    X_train, X_test, ys_train, ys_test, yd_train, yd_test = train_test_split(
        X, y_sys, y_dia, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, ys_train, ys_test, yd_train, yd_test, scaler
