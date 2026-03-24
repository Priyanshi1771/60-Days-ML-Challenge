"""Day 14: Drug Response Prediction — Data Pipeline"""
import logging, numpy as np, pandas as pd, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import config

logger = logging.getLogger(__name__)

# Word pools for generating realistic reviews
_POS_WORDS = [
    "excellent works great helped amazing relief effective wonderful life-changing",
    "fantastic improvement cleared cured recommend miracle perfect best satisfied",
    "reduced pain gone better improved manageable tolerable quality sleep energy",
    "no side effects minimal issues well-tolerated easy comfortable positive",
].join(" ").split() if False else (
    "excellent works great helped amazing relief effective wonderful life-changing "
    "fantastic improvement cleared cured recommend miracle perfect best satisfied "
    "reduced pain gone better improved manageable tolerable quality sleep energy "
    "no-side-effects minimal-issues well-tolerated easy comfortable positive"
).split()

_NEG_WORDS = (
    "terrible horrible worst awful useless ineffective dangerous toxic waste "
    "nausea vomiting headache dizzy fatigue insomnia weight-gain rash itching "
    "worse worsened failed nothing painful unbearable miserable suffering "
    "side-effects severe allergic reaction hospitalized emergency discontinue"
).split()

_NEU_WORDS = (
    "took prescribed doctor weeks months daily medication pill tablet dose "
    "started began condition diagnosed treatment therapy chronic symptoms "
    "blood-pressure cholesterol diabetes anxiety depression pain arthritis"
).split()

_CONDITIONS = [
    "Depression", "Anxiety", "Diabetes Type 2", "High Blood Pressure",
    "Pain", "Arthritis", "Insomnia", "Migraine", "Asthma", "Allergies",
    "GERD", "Acne", "Hypothyroidism", "High Cholesterol", "ADHD",
    "Bipolar Disorder", "Obesity", "UTI", "Fibromyalgia", "COPD"
]

_DRUGS = [
    "Metformin", "Lisinopril", "Atorvastatin", "Amlodipine", "Omeprazole",
    "Sertraline", "Gabapentin", "Losartan", "Ibuprofen", "Acetaminophen",
    "Escitalopram", "Duloxetine", "Prednisone", "Amoxicillin", "Tramadol",
    "Hydrochlorothiazide", "Levothyroxine", "Montelukast", "Trazodone", "Cyclobenzaprine",
    "Meloxicam", "Pantoprazole", "Bupropion", "Venlafaxine", "Alprazolam"
]


def load_data():
    logger.info("=" * 60)
    logger.info("LOADING DRUG REVIEW DATASET")
    logger.info("=" * 60)

    rng = np.random.RandomState(config.RANDOM_SEED)
    n = 6000

    conditions = rng.choice(_CONDITIONS, n)
    drugs = rng.choice(_DRUGS, n)
    ratings = rng.choice(np.arange(1, 11), n, p=[0.08, 0.05, 0.06, 0.05, 0.07,
                                                    0.06, 0.08, 0.15, 0.18, 0.22])
    useful = (rng.exponential(15, n) * (ratings / 5)).clip(0, 200).astype(int)

    # Generate reviews correlated with rating
    reviews = []
    for i in range(n):
        r = ratings[i]
        length = rng.randint(15, 80)
        words = []

        if r >= 7:
            words += list(rng.choice(_POS_WORDS, int(length * 0.5)))
            words += list(rng.choice(_NEU_WORDS, int(length * 0.3)))
            words += list(rng.choice(_NEG_WORDS, int(length * 0.05)))
        elif r <= 3:
            words += list(rng.choice(_NEG_WORDS, int(length * 0.5)))
            words += list(rng.choice(_NEU_WORDS, int(length * 0.3)))
            words += list(rng.choice(_POS_WORDS, int(length * 0.05)))
        else:
            words += list(rng.choice(_NEU_WORDS, int(length * 0.4)))
            words += list(rng.choice(_POS_WORDS, int(length * 0.2)))
            words += list(rng.choice(_NEG_WORDS, int(length * 0.2)))

        # Add condition/drug name sometimes
        if rng.random() > 0.3:
            words.insert(0, conditions[i].lower().replace(" ", "-"))
        if rng.random() > 0.5:
            words.insert(rng.randint(0, len(words)), drugs[i].lower())

        rng.shuffle(words)
        reviews.append(" ".join(words[:length]))

    df = pd.DataFrame({
        "drug": drugs, "condition": conditions, "review": reviews,
        "rating": ratings.astype(np.float32), "usefulCount": useful
    })

    logger.info(f"Generated {n} drug reviews | Rating range: [1, 10]")
    logger.info(f"Drugs: {len(_DRUGS)} unique | Conditions: {len(_CONDITIONS)} unique")
    return df


def explore_data(df):
    logger.info("-" * 60)
    logger.info("EDA")
    logger.info("-" * 60)

    logger.info(f"Shape: {df.shape}")
    logger.info(f"Rating: mean={df['rating'].mean():.2f}, median={df['rating'].median():.0f}")
    logger.info(f"Review length: mean={df['review'].str.len().mean():.0f} chars")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Rating distribution
    axes[0,0].hist(df["rating"], bins=10, color='#AB47BC', edgecolor='white', alpha=0.85, rwidth=0.85)
    axes[0,0].set_title("💊 Drug Rating Distribution", fontweight='bold')
    axes[0,0].set_xlabel("Rating (1-10)"); axes[0,0].spines[['top','right']].set_visible(False)

    # Top conditions
    top_cond = df["condition"].value_counts().head(10)
    axes[0,1].barh(top_cond.index, top_cond.values, color='#4FC3F7', edgecolor='white')
    axes[0,1].set_title("🩺 Top 10 Conditions", fontweight='bold')
    axes[0,1].invert_yaxis(); axes[0,1].spines[['top','right']].set_visible(False)

    # Top drugs
    top_drugs = df["drug"].value_counts().head(10)
    axes[0,2].barh(top_drugs.index, top_drugs.values, color='#66BB6A', edgecolor='white')
    axes[0,2].set_title("💉 Top 10 Drugs", fontweight='bold')
    axes[0,2].invert_yaxis(); axes[0,2].spines[['top','right']].set_visible(False)

    # Review length vs rating
    df["_len"] = df["review"].str.len()
    axes[1,0].scatter(df["_len"], df["rating"], alpha=0.1, s=8, color='#FF7043', rasterized=True)
    axes[1,0].set_title("📝 Review Length vs Rating", fontweight='bold')
    axes[1,0].set_xlabel("Review Length (chars)"); axes[1,0].set_ylabel("Rating")
    axes[1,0].spines[['top','right']].set_visible(False)

    # Useful count vs rating
    axes[1,1].scatter(df["usefulCount"], df["rating"], alpha=0.1, s=8, color='#AB47BC', rasterized=True)
    axes[1,1].set_title("👍 Useful Count vs Rating", fontweight='bold')
    axes[1,1].set_xlabel("Useful Count"); axes[1,1].set_ylabel("Rating")
    axes[1,1].spines[['top','right']].set_visible(False)

    # Rating by condition (boxplot for top 8)
    top8 = df["condition"].value_counts().head(8).index
    data_box = [df[df["condition"] == c]["rating"].values for c in top8]
    bp = axes[1,2].boxplot(data_box, labels=[c[:12] for c in top8], patch_artist=True, vert=True)
    for patch, color in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, 8))):
        patch.set_facecolor(color)
    axes[1,2].set_title("📊 Rating by Condition", fontweight='bold')
    axes[1,2].tick_params(axis='x', rotation=35)
    axes[1,2].spines[['top','right']].set_visible(False)

    df.drop("_len", axis=1, inplace=True)
    plt.suptitle("Drug Review Dataset — Exploratory Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/01_eda.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 01_eda.png")


def extract_features(df):
    """
    TF-IDF: converts raw text into numerical features.
    Term Frequency × Inverse Document Frequency
    - High TF-IDF = word is important in THIS review but rare overall
    - "excellent" in a positive review → high TF-IDF
    - "the" everywhere → low TF-IDF (filtered out)
    """
    logger.info("-" * 60)
    logger.info("TF-IDF FEATURE EXTRACTION")
    logger.info("-" * 60)

    # Clean text minimally
    df["review_clean"] = df["review"].str.lower().apply(lambda x: re.sub(r'[^a-z\s-]', '', x))

    # Engineered numeric features
    df["review_length"] = df["review"].str.len().astype(np.float32)
    df["word_count"] = df["review"].str.split().str.len().astype(np.float32)
    df["condition_freq"] = df.groupby("condition")["condition"].transform("count").astype(np.float32)
    df["drug_freq"] = df.groupby("drug")["drug"].transform("count").astype(np.float32)

    # Split BEFORE fitting TF-IDF (no data leakage!)
    y = df[config.TARGET_NAME].values.astype(np.float32)
    X_text = df["review_clean"].values
    X_num = df[config.NUMERIC_FEATURES].values.astype(np.float32)

    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_text, X_num, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)

    # TF-IDF fit on train only
    tfidf = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        dtype=np.float32
    )
    X_tfidf_train = tfidf.fit_transform(X_text_train)  # sparse matrix
    X_tfidf_test = tfidf.transform(X_text_test)

    # Scale numeric features (fit on train)
    scaler = StandardScaler()
    X_num_train_s = scaler.fit_transform(X_num_train).astype(np.float32)
    X_num_test_s = scaler.transform(X_num_test).astype(np.float32)

    # Combine: sparse TF-IDF + dense numeric → sparse
    X_train = hstack([X_tfidf_train, csr_matrix(X_num_train_s)], format='csr')
    X_test = hstack([X_tfidf_test, csr_matrix(X_num_test_s)], format='csr')

    n_tfidf = X_tfidf_train.shape[1]
    n_total = X_train.shape[1]
    logger.info(f"TF-IDF features: {n_tfidf} | Numeric: {len(config.NUMERIC_FEATURES)} | Total: {n_total}")
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Sparsity: {1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.4f}")

    # Plot top TF-IDF features
    _plot_top_tfidf(tfidf, X_tfidf_train, y_train)

    return X_train, X_test, y_train, y_test, tfidf, scaler


def _plot_top_tfidf(tfidf, X_tfidf_train, y_train):
    """Show which words TF-IDF considers most important."""
    feature_names = np.array(tfidf.get_feature_names_out())
    mean_tfidf = np.array(X_tfidf_train.mean(axis=0)).ravel()
    top30_idx = np.argsort(mean_tfidf)[::-1][:30]

    # Correlation of each top word with rating
    corrs = []
    for idx in top30_idx:
        col = np.array(X_tfidf_train[:, idx].todense()).ravel()
        r = np.corrcoef(col, y_train)[0, 1]
        corrs.append(r)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Top words by frequency
    ax = axes[0]
    colors = ['#66BB6A' if c > 0.05 else '#EF5350' if c < -0.05 else '#FFB74D' for c in corrs]
    ax.barh(range(30), mean_tfidf[top30_idx], color=colors, edgecolor='white')
    ax.set_yticks(range(30))
    ax.set_yticklabels(feature_names[top30_idx], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean TF-IDF Score")
    ax.set_title("📝 Top 30 TF-IDF Features\n(Green=↑rating | Red=↓rating | Yellow=neutral)", fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    # Word correlation with rating
    ax = axes[1]
    sorted_corr_idx = np.argsort(corrs)
    colors2 = ['#66BB6A' if corrs[i] > 0 else '#EF5350' for i in sorted_corr_idx]
    ax.barh(range(30), [corrs[i] for i in sorted_corr_idx], color=colors2, edgecolor='white')
    ax.set_yticks(range(30))
    ax.set_yticklabels([feature_names[top30_idx[i]] for i in sorted_corr_idx], fontsize=8)
    ax.axvline(0, color='gray', lw=1, linestyle='--')
    ax.set_xlabel("Correlation with Rating")
    ax.set_title("🔬 Word Sentiment Signal\n(Green=positive words | Red=negative words)", fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/02_tfidf_features.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 02_tfidf_features.png")
