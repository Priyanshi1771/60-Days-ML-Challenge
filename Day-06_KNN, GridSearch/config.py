"""
=============================================================================
 Day 6: Kidney Disease Prediction — Configuration
=============================================================================
 Dataset  : UCI Chronic Kidney Disease (400 samples, 24 features)
 Models   : KNN (primary), + comparison baselines (LR, SVM, RF, DT)
 Focus    : GridSearchCV hyperparameter tuning deep-dive
=============================================================================
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Data ─────────────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
N_SPLITS = 10  # Stratified K-Fold

# ─── UCI CKD Dataset Info ─────────────────────────────────────────────────────
# 24 features (11 numeric + 13 nominal) → 2 classes: CKD vs Not-CKD
# Heavy missing values (~35% of cells), mixed types
NUMERIC_FEATURES = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo",
    "pcv", "wc", "rc"
]
CATEGORICAL_FEATURES = [
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"
]
TARGET_NAME = "classification"
CLASS_NAMES = ["Not CKD", "CKD"]

# ─── KNN Hyperparameter Grid (exhaustive) ────────────────────────────────────
KNN_PARAM_GRID = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
    "p": [1, 2, 3],  # Power parameter for Minkowski
}

# ─── Baseline Model Params ────────────────────────────────────────────────────
LR_C_RANGE = [0.01, 0.1, 1.0, 10.0]
SVM_C_RANGE = [0.1, 1.0, 10.0]
RF_N_ESTIMATORS = [50, 100, 200]

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
