"""
=============================================================================
 Day 7: Stroke Risk Prediction — Data Pipeline
=============================================================================
 KEY CHALLENGE: Extreme class imbalance (~95% No Stroke, ~5% Stroke)
 This is the most imbalanced dataset in the challenge so far.
=============================================================================
"""
import logging, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import config

logger = logging.getLogger(__name__)


def load_data():
    logger.info("=" * 60)
    logger.info("LOADING STROKE PREDICTION DATASET")
    logger.info("=" * 60)
    try:
        import kaggle
        df = pd.read_csv(f"{config.DATA_DIR}/healthcare-dataset-stroke-data.csv")
    except:
        logger.info("Generating realistic synthetic stroke dataset...")
        df = _generate_fallback_data()
    logger.info(f"Dataset shape: {df.shape}")
    return df


def _generate_fallback_data():
    """Realistic stroke data: ~5% positive rate, mixed features, missing BMI."""
    np.random.seed(config.RANDOM_SEED)
    n = 5110
    n_stroke = int(n * 0.049)  # ~249 stroke cases
    n_healthy = n - n_stroke

    def gen(n_h, n_s):
        age = np.concatenate([np.random.normal(42, 18, n_h).clip(1, 82), np.random.normal(68, 12, n_s).clip(30, 90)])
        hyp = np.concatenate([np.random.binomial(1, 0.08, n_h), np.random.binomial(1, 0.42, n_s)])
        hd = np.concatenate([np.random.binomial(1, 0.04, n_h), np.random.binomial(1, 0.22, n_s)])
        glucose = np.concatenate([np.random.normal(95, 35, n_h).clip(50, 250), np.random.normal(140, 55, n_s).clip(55, 300)])
        bmi = np.concatenate([np.random.normal(27, 6, n_h).clip(12, 55), np.random.normal(31, 7, n_s).clip(15, 60)])
        gender = np.random.choice(["Male", "Female", "Other"], n, p=[0.41, 0.585, 0.005])
        married = np.concatenate([np.random.choice(["Yes", "No"], n_h, p=[0.55, 0.45]),
                                  np.random.choice(["Yes", "No"], n_s, p=[0.82, 0.18])])
        work = np.concatenate([np.random.choice(["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                                                 n_h, p=[0.55, 0.15, 0.12, 0.13, 0.05]),
                               np.random.choice(["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                                                 n_s, p=[0.52, 0.25, 0.18, 0.03, 0.02])])
        residence = np.random.choice(["Urban", "Rural"], n, p=[0.51, 0.49])
        smoking = np.concatenate([np.random.choice(["never smoked", "formerly smoked", "smokes", "Unknown"],
                                                    n_h, p=[0.37, 0.17, 0.15, 0.31]),
                                  np.random.choice(["never smoked", "formerly smoked", "smokes", "Unknown"],
                                                    n_s, p=[0.22, 0.28, 0.30, 0.20])])
        stroke = np.concatenate([np.zeros(n_h), np.ones(n_s)]).astype(int)
        return pd.DataFrame({"gender": gender, "age": age, "hypertension": hyp, "heart_disease": hd,
                              "ever_married": married, "work_type": work, "Residence_type": residence,
                              "avg_glucose_level": glucose, "bmi": bmi, "smoking_status": smoking,
                              "stroke": stroke})

    df = gen(n_healthy, n_stroke)
    # Inject ~3.5% missing BMI
    miss_idx = np.random.choice(n, size=int(n * 0.035), replace=False)
    df.loc[miss_idx, "bmi"] = np.nan
    # Inject ~2% label noise for realism
    noise_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
    df.loc[noise_idx, "stroke"] = 1 - df.loc[noise_idx, "stroke"]
    return df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)


def explore_data(df):
    logger.info("-" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("-" * 60)
    logger.info(f"Samples: {len(df)} | Features: {df.shape[1]-1}")
    logger.info(f"Missing: {df.isnull().sum().sum()} ({df.isnull().sum().sum()/df.size*100:.2f}%)")

    cd = df[config.TARGET_NAME].value_counts()
    for cls, cnt in cd.items():
        logger.info(f"  Class {cls}: {cnt} ({cnt/len(df)*100:.1f}%)")
    logger.info(f"  ⚠️  Imbalance ratio: {cd.values[0]/cd.values[1]:.1f}:1")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#66BB6A", "#EF5350"]
    axes[0].bar(config.CLASS_NAMES, cd.values, color=colors, edgecolor="white")
    for i, (cnt, name) in enumerate(zip(cd.values, config.CLASS_NAMES)):
        axes[0].text(i, cnt + 20, f'{cnt}\n({cnt/len(df)*100:.1f}%)', ha='center', fontweight='bold')
    axes[0].set_title("⚡ Class Distribution (Extreme Imbalance)", fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)

    for stroke_val, color, label in [(0, "#4FC3F7", "No Stroke"), (1, "#EF5350", "Stroke")]:
        subset = df[df["stroke"] == stroke_val]["age"].dropna()
        axes[1].hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
    axes[1].set_title("🧠 Age Distribution by Stroke Status", fontweight='bold')
    axes[1].set_xlabel("Age")
    axes[1].legend()
    axes[1].spines[['top', 'right']].set_visible(False)

    for stroke_val, color, label in [(0, "#4FC3F7", "No Stroke"), (1, "#EF5350", "Stroke")]:
        subset = df[df["stroke"] == stroke_val]["avg_glucose_level"].dropna()
        axes[2].hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor='white')
    axes[2].set_title("💉 Glucose Distribution by Stroke Status", fontweight='bold')
    axes[2].set_xlabel("Average Glucose Level")
    axes[2].legend()
    axes[2].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_eda_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_eda_overview.png")

    # Correlation heatmap for numeric features
    fig, ax = plt.subplots(figsize=(8, 6))
    num_cols = config.NUMERIC_FEATURES + [config.TARGET_NAME]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, square=True)
    ax.set_title("🔬 Feature Correlation with Stroke", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_correlation.png")
    return cd


def preprocess_and_split(df):
    logger.info("-" * 60)
    logger.info("PREPROCESSING & SPLITTING")
    logger.info("-" * 60)
    df = df.copy()
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    # Encode categoricals
    label_encoders = {}
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str).str.strip()
            le.fit(df[col])
            df[col] = le.transform(df[col])
            label_encoders[col] = le
    logger.info(f"Encoded {len(label_encoders)} categorical features")

    # Separate
    feature_cols = [c for c in df.columns if c != config.TARGET_NAME]
    X = df[feature_cols].values.astype(np.float64)
    y = df[config.TARGET_NAME].values.astype(int)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y)
    logger.info(f"Split: Train={len(y_train)} | Test={len(y_test)}")
    logger.info(f"  Train stroke rate: {y_train.mean():.4f}")
    logger.info(f"  Test stroke rate:  {y_test.mean():.4f}")

    # Impute BMI (median, train only)
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scale (for baselines; XGBoost doesn't need it but it doesn't hurt)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Imputed + scaled (fit on train only)")

    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler, label_encoders, feature_cols
