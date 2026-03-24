"""Day 15: BMI Prediction — Data Pipeline"""
import logging, numpy as np, pandas as pd
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
    logger.info("LOADING OBESITY / BMI DATASET")
    logger.info("=" * 60)

    rng = np.random.RandomState(config.RANDOM_SEED)
    n = 5500

    age = rng.normal(42, 14, n).clip(18, 75).astype(np.float32)
    gender = rng.binomial(1, 0.48, n).astype(np.float32)
    height = (rng.normal(170, 10, n) - gender * 12 + rng.normal(0, 3, n)).clip(145, 200).astype(np.float32)
    
    # Weight is the hidden key to BMI — generate realistic distribution
    bmi_true = rng.normal(26.5, 5.5, n).clip(15, 50).astype(np.float32)
    weight = (bmi_true * (height / 100) ** 2).clip(40, 170).astype(np.float32)

    # Body measurements correlated with BMI
    waist = (bmi_true * 2.8 + rng.normal(0, 4, n) - gender * 5).clip(55, 160).astype(np.float32)
    hip = (waist * 0.95 + rng.normal(8, 3, n) + gender * 8).clip(60, 155).astype(np.float32)
    neck = (bmi_true * 0.5 + rng.normal(28, 2, n) - gender * 3).clip(25, 50).astype(np.float32)
    chest = (bmi_true * 1.8 + rng.normal(55, 5, n) - gender * 8).clip(60, 140).astype(np.float32)
    abdomen = (waist * 1.02 + rng.normal(0, 3, n)).clip(55, 165).astype(np.float32)
    thigh = (bmi_true * 0.9 + rng.normal(40, 4, n) + gender * 4).clip(30, 80).astype(np.float32)
    knee = (rng.normal(37, 3, n) + bmi_true * 0.15).clip(28, 50).astype(np.float32)
    ankle = (rng.normal(22, 2, n) + bmi_true * 0.08).clip(16, 32).astype(np.float32)
    bicep = (bmi_true * 0.5 + rng.normal(22, 3, n) - gender * 3).clip(18, 45).astype(np.float32)
    forearm = (rng.normal(27, 2, n) + bmi_true * 0.12 - gender * 2).clip(20, 38).astype(np.float32)
    wrist = (rng.normal(17, 1.2, n) + bmi_true * 0.05 - gender * 1.5).clip(13, 22).astype(np.float32)

    # Lifestyle
    calories = (1800 + bmi_true * 30 + rng.normal(0, 300, n) - gender * 200).clip(1000, 5000).astype(np.float32)
    protein = (calories * 0.15 / 4 + rng.normal(0, 10, n)).clip(30, 250).astype(np.float32)
    carbs = (calories * 0.50 / 4 + rng.normal(0, 20, n)).clip(80, 500).astype(np.float32)
    fat = (calories * 0.35 / 9 + rng.normal(0, 8, n)).clip(20, 200).astype(np.float32)
    exercise = rng.exponential(3.0, n).clip(0, 15).astype(np.float32)
    sedentary = (rng.normal(7, 2.5, n) + bmi_true * 0.1).clip(1, 16).astype(np.float32)
    sleep = rng.normal(7, 1.2, n).clip(3, 12).astype(np.float32)
    water = rng.normal(2.0, 0.6, n).clip(0.3, 5).astype(np.float32)
    alcohol = rng.exponential(2, n).clip(0, 25).astype(np.float32)
    smoker = rng.binomial(1, 0.18, n).astype(np.float32)

    X = np.column_stack([
        age, gender, height, weight, waist, hip, neck, chest, abdomen,
        thigh, knee, ankle, bicep, forearm, wrist,
        calories, protein, carbs, fat, exercise, sedentary, sleep, water, alcohol, smoker
    ])

    # Target BMI with noise (don't leak weight/height directly)
    y = (bmi_true + rng.normal(0, 0.8, n)).clip(14, 52).astype(np.float32)

    logger.info(f"Generated {n} patients | {X.shape[1]} raw features")
    logger.info(f"BMI: mean={y.mean():.1f}, std={y.std():.1f}, range=[{y.min():.1f}, {y.max():.1f}]")
    return X, y


def explore_data(X, y):
    logger.info("-" * 60)
    logger.info("EDA")
    logger.info("-" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # BMI distribution with WHO categories
    ax = axes[0, 0]
    ax.hist(y, bins=50, color='#FF7043', edgecolor='white', alpha=0.85)
    for thresh, label, color in [(18.5, 'Under', '#4FC3F7'), (25, 'Normal', '#66BB6A'),
                                   (30, 'Overweight', '#FFB74D'), (35, 'Obese I', '#FF7043'),
                                   (40, 'Obese II', '#EF5350')]:
        ax.axvline(thresh, color=color, lw=2, linestyle='--', alpha=0.7)
        ax.text(thresh + 0.3, ax.get_ylim()[1] * 0.9, label, fontsize=7, color=color, rotation=90)
    ax.set_title("⚖️ BMI Distribution + WHO Categories", fontweight='bold')
    ax.set_xlabel("BMI (kg/m²)"); ax.spines[['top', 'right']].set_visible(False)

    # Top correlated features
    corrs = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
    top5 = np.argsort(np.abs(corrs))[::-1][:5]
    for ax_i, fi in enumerate(top5):
        ax = axes.ravel()[ax_i + 1]
        ax.scatter(X[:, fi], y, alpha=0.08, s=6, color='#4FC3F7', rasterized=True)
        r = corrs[fi]
        ax.set_title(f"🔬 {config.RAW_FEATURES[fi]} (r={r:.3f})", fontweight='bold', fontsize=10)
        ax.set_ylabel("BMI"); ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle("BMI Prediction — Target & Top Raw Predictors", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_eda.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_eda.png")


def engineer_interactions(X, feature_names):
    """
    Create interaction features: ratios and products of correlated pairs.
    Interaction terms capture COMBINED effects that individual features miss.
    Example: waist/hip ratio is a better obesity predictor than waist or hip alone.
    """
    logger.info("-" * 60)
    logger.info("FEATURE ENGINEERING — Interaction Terms")
    logger.info("-" * 60)

    X_new = X.copy()
    new_names = list(feature_names)
    interaction_data = []

    for f1_name, f2_name in config.INTERACTION_PAIRS:
        i1 = feature_names.index(f1_name)
        i2 = feature_names.index(f2_name)
        col1, col2 = X[:, i1], X[:, i2]

        # Ratio feature
        ratio = np.divide(col1, col2, where=col2 != 0, out=np.zeros_like(col1))
        ratio_name = f"{f1_name}/{f2_name}"
        X_new = np.column_stack([X_new, ratio.astype(np.float32)])
        new_names.append(ratio_name)

        # Product feature
        product = (col1 * col2).astype(np.float32)
        prod_name = f"{f1_name}×{f2_name}"
        X_new = np.column_stack([X_new, product])
        new_names.append(prod_name)

        interaction_data.append((ratio_name, prod_name, f1_name, f2_name))
        logger.info(f"  Created: {ratio_name:35s} + {prod_name}")

    n_new = X_new.shape[1] - X.shape[1]
    logger.info(f"\n  Raw features: {X.shape[1]} → With interactions: {X_new.shape[1]} (+{n_new} new)")

    return X_new, new_names, interaction_data


def compare_with_without_interactions(X_raw, X_inter, y, raw_names, inter_names):
    """Visualize which interaction features add predictive value."""
    from sklearn.ensemble import RandomForestRegressor

    # Quick RF on raw vs raw+interactions
    rng = np.random.RandomState(config.RANDOM_SEED)
    idx = rng.permutation(len(y))
    n_train = int(len(y) * 0.8)
    tr, te = idx[:n_train], idx[n_train:]

    rf_raw = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=config.RANDOM_SEED, n_jobs=-1)
    rf_raw.fit(X_raw[tr], y[tr])
    r2_raw = 1 - np.sum((y[te] - rf_raw.predict(X_raw[te])) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)

    rf_inter = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=config.RANDOM_SEED, n_jobs=-1)
    rf_inter.fit(X_inter[tr], y[tr])
    r2_inter = 1 - np.sum((y[te] - rf_inter.predict(X_inter[te])) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)

    logger.info(f"\n  Quick comparison (RF, 100 trees):")
    logger.info(f"    Raw features only:      R² = {r2_raw:.4f}  ({len(raw_names)} features)")
    logger.info(f"    With interactions:       R² = {r2_inter:.4f}  ({len(inter_names)} features)")
    logger.info(f"    Improvement:             +{(r2_inter - r2_raw):.4f}")

    # Feature importance comparison
    fi_inter = rf_inter.feature_importances_
    sorted_idx = np.argsort(fi_inter)[::-1][:20]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # (0) Top features with interactions
    colors = ['#FF7043' if i >= len(raw_names) else '#4FC3F7' for i in sorted_idx]
    axes[0].barh(range(20), fi_inter[sorted_idx], color=colors, edgecolor='white')
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels([inter_names[i] for i in sorted_idx], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Feature Importance (Gini)")
    axes[0].set_title("🔧 Top 20 Features (Orange = Interaction Terms!)", fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)

    # (1) R² improvement bar
    axes[1].bar(["Raw Features\nOnly", "With\nInteractions"], [r2_raw, r2_inter],
                color=['#4FC3F7', '#FF7043'], edgecolor='white', width=0.5)
    axes[1].set_ylabel("R² Score")
    axes[1].set_title(f"📈 Impact of Interaction Features\n(+{(r2_inter-r2_raw):.4f} R² improvement)", fontweight='bold')
    for i, v in enumerate([r2_raw, r2_inter]):
        axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold', fontsize=12)
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_interaction_impact.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_interaction_impact.png")

    return r2_raw, r2_inter


def preprocess_and_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    logger.info(f"Split: Train={X_train.shape[0]} | Test={X_test.shape[0]} | Features={X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, scaler
