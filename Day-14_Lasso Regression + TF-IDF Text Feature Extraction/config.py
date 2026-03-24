"""Day 14: Drug Response Prediction — Config"""
import matplotlib
matplotlib.use('Agg')
import os, torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TF-IDF settings
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)    # unigrams + bigrams
TFIDF_MIN_DF = 3              # ignore words appearing < 3 times
TFIDF_MAX_DF = 0.95           # ignore words in > 95% of docs

# Lasso alpha sweep
LASSO_ALPHAS = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

# Numeric features from structured data
NUMERIC_FEATURES = ["usefulCount", "condition_freq", "drug_freq", "review_length", "word_count"]
TARGET_NAME = "rating"

# GPU Neural Net
NN_HIDDEN = [256, 128]
NN_EPOCHS = 60
NN_BATCH = 256
NN_LR = 1e-3

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
