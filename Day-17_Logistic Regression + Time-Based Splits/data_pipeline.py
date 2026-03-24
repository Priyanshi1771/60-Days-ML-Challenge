"""Day 17: Hospital Readmission Risk — Data Pipeline"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import config

logger = logging.getLogger(__name__)


def load_data():
    logger.info("=" * 60)
    logger.info("LOADING HOSPITAL READMISSION DATASET")
    logger.info("=" * 60)

    rng = np.random.RandomState(config.RANDOM_SEED)
    n = config.N_PATIENTS

    # Admission dates over 4 years (temporal ordering is KEY)
    start = pd.Timestamp("2018-01-01")
    admission_dates = pd.to_datetime(
        start + pd.to_timedelta(rng.randint(0, 365 * config.TIME_SPAN_YEARS, n), unit='D'))
    admission_dates = np.sort(admission_dates)  # chronological order

    age = rng.normal(65, 14, n).clip(18, 95).astype(np.float32)
    gender = rng.binomial(1, 0.46, n).astype(np.float32)
    prev_admits = rng.poisson(1.5, n).clip(0, 15).astype(np.float32)
    los = rng.lognormal(1.2, 0.7, n).clip(1, 30).astype(np.float32)
    n_diag = rng.poisson(5, n).clip(1, 20).astype(np.float32)
    n_proc = rng.poisson(2, n).clip(0, 12).astype(np.float32)
    n_meds = rng.poisson(8, n).clip(1, 25).astype(np.float32)
    n_labs = rng.poisson(10, n).clip(1, 40).astype(np.float32)
    hba1c = rng.choice([0, 1, 2, 3], n, p=[0.55, 0.15, 0.15, 0.15]).astype(np.float32)  # 0=not measured
    discharge = rng.choice([0, 1, 2, 3], n, p=[0.60, 0.20, 0.12, 0.08]).astype(np.float32)
    admission_src = rng.choice([0, 1, 2, 3], n, p=[0.45, 0.30, 0.15, 0.10]).astype(np.float32)
    diag_group = rng.choice(np.arange(8), n).astype(np.float32)
    outpatient = rng.poisson(1, n).clip(0, 10).astype(np.float32)
    er_visits = rng.poisson(0.5, n).clip(0, 8).astype(np.float32)
    inpatient = rng.poisson(0.3, n).clip(0, 5).astype(np.float32)
    insulin = rng.binomial(1, 0.25, n).astype(np.float32)
    metformin = rng.binomial(1, 0.20, n).astype(np.float32)
    diabetic = ((hba1c >= 2) | (insulin == 1)).astype(np.float32)
    comorbidity = (prev_admits * 0.3 + n_diag * 0.2 + age * 0.02 + rng.normal(0, 0.5, n)).clip(0, 10).astype(np.float32)
    payer = rng.choice([0, 1, 2, 3], n, p=[0.45, 0.25, 0.20, 0.10]).astype(np.float32)

    X = np.column_stack([
        age, gender, prev_admits, los, n_diag, n_proc, n_meds, n_labs, hba1c,
        discharge, admission_src, diag_group, outpatient, er_visits, inpatient,
        insulin, metformin, diabetic, comorbidity, payer
    ])

    # Target: readmission probability influenced by risk factors
    # Also includes TEMPORAL DRIFT: readmission policies tightened over time
    year_idx = (pd.DatetimeIndex(admission_dates) - start).days / 365.25
    temporal_effect = -0.1 * year_idx  # later years have lower readmission (policy improvement)

    logit = (
        -2.5
        + 0.015 * age
        + 0.25 * prev_admits
        + 0.04 * los
        + 0.08 * n_diag
        + 0.1 * er_visits
        + 0.15 * inpatient
        + 0.12 * comorbidity
        - 0.05 * outpatient
        + 0.2 * (hba1c >= 2).astype(float)
        + 0.3 * (discharge >= 2).astype(float)
        + temporal_effect
        + rng.normal(0, 0.3, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(np.int8)

    df = pd.DataFrame(X, columns=config.FEATURE_NAMES)
    df["admission_date"] = admission_dates
    df[config.TARGET_NAME] = y

    logger.info(f"Generated {n} admissions over {config.TIME_SPAN_YEARS} years")
    logger.info(f"Readmission rate: {y.mean():.3f} ({y.sum()}/{n})")
    logger.info(f"Date range: {admission_dates.min().date()} → {admission_dates.max().date()}")
    return df


def explore_data(df):
    logger.info("-" * 60)
    logger.info("EDA + TEMPORAL ANALYSIS")
    logger.info("-" * 60)

    y = df[config.TARGET_NAME].values
    logger.info(f"Not readmitted: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    logger.info(f"Readmitted:     {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Class distribution
    axes[0,0].bar(config.CLASS_NAMES, [(y==0).sum(), (y==1).sum()],
                  color=['#66BB6A', '#EF5350'], edgecolor='white')
    axes[0,0].set_title("🏥 Readmission Distribution", fontweight='bold')
    axes[0,0].spines[['top','right']].set_visible(False)

    # Readmission rate over time (temporal drift!)
    df["_quarter"] = df["admission_date"].dt.to_period("Q")
    quarterly = df.groupby("_quarter")[config.TARGET_NAME].mean()
    axes[0,1].plot(range(len(quarterly)), quarterly.values, 'o-', color='#EF5350', lw=2)
    axes[0,1].set_xlabel("Quarter"); axes[0,1].set_ylabel("Readmission Rate")
    axes[0,1].set_title("📉 Readmission Rate Over Time (DRIFT!)\n(Policies improved → lower rate in recent years)", fontweight='bold', fontsize=10)
    axes[0,1].spines[['top','right']].set_visible(False)
    axes[0,1].grid(alpha=0.3)

    # Key predictors
    for ax_i, (feat, title) in enumerate([
        ("n_prev_admissions", "🔄 Previous Admissions"),
        ("comorbidity_score", "🩺 Comorbidity Score"),
        ("los_days", "🛏️ Length of Stay"),
        ("age", "👤 Age")]):
        ax = axes.ravel()[ax_i + 2]
        for cls, color, label in [(0, '#66BB6A', 'Not Readmitted'), (1, '#EF5350', 'Readmitted')]:
            subset = df[df[config.TARGET_NAME] == cls][feat]
            ax.hist(subset, bins=20, alpha=0.6, color=color, label=label, edgecolor='white')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)

    df.drop("_quarter", axis=1, inplace=True)
    plt.suptitle("Hospital Readmission — Exploratory Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_eda.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_eda.png")


def time_based_split(df):
    """
    TEMPORAL SPLIT: train on PAST, test on FUTURE.
    This is how hospital models would actually be deployed:
    - Train on 2018-2020 admissions
    - Deploy to predict 2021 admissions
    
    Random split would leak future information (e.g., policy changes in 2021
    leaking into 2018 training data).
    """
    logger.info("-" * 60)
    logger.info("TIME-BASED SPLIT (NO RANDOM SHUFFLE!)")
    logger.info("-" * 60)

    df_sorted = df.sort_values("admission_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.75)  # 75% train, 25% test (temporal)

    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    train_cutoff = train_df["admission_date"].max()
    test_start = test_df["admission_date"].min()

    logger.info(f"  Train: {len(train_df)} admissions ({train_df['admission_date'].min().date()} → {train_cutoff.date()})")
    logger.info(f"  Test:  {len(test_df)} admissions ({test_start.date()} → {test_df['admission_date'].max().date()})")
    logger.info(f"  ⚠️  NO overlap — model never sees future data during training!")

    # Show temporal drift between splits
    train_rate = train_df[config.TARGET_NAME].mean()
    test_rate = test_df[config.TARGET_NAME].mean()
    logger.info(f"  Train readmission rate: {train_rate:.4f}")
    logger.info(f"  Test readmission rate:  {test_rate:.4f}")
    logger.info(f"  Drift: {abs(test_rate - train_rate):.4f} ({'⚠️ significant' if abs(test_rate - train_rate) > 0.02 else '✅ minimal'})")

    return train_df, test_df


def random_split_for_comparison(df):
    """Random split (the WRONG way for temporal data) — we compare both."""
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=config.RANDOM_SEED,
                                          stratify=df[config.TARGET_NAME])
    return train_df, test_df


def prepare_features(train_df, test_df):
    """Extract X, y and scale features (fit on train only)."""
    feat_cols = config.FEATURE_NAMES

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df[config.TARGET_NAME].values.astype(np.int8)
    X_test = test_df[feat_cols].values.astype(np.float32)
    y_test = test_df[config.TARGET_NAME].values.astype(np.int8)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test, scaler
