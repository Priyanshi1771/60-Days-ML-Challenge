"""
=============================================================================
 Day 5: Thyroid Disease Classification — Configuration
=============================================================================
 Dataset  : UCI Thyroid Disease (New Thyroid)
 Models   : Gaussian Naive Bayes, Multinomial NB, Voting Ensemble
 Focus    : Ensemble voting classifiers (hard + soft voting)
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
N_SPLITS = 10  # Stratified K-Fold splits

# ─── UCI New Thyroid Dataset Info ─────────────────────────────────────────────
# Classes: 1 = Normal, 2 = Hyperthyroid, 3 = Hypothyroid
# Features: T3-resin uptake, Total Serum thyroxin, Total serum triiodothyronine,
#           TSH (Thyroid stimulating hormone), Max diff of TSH after injection of TRH
FEATURE_NAMES = [
    "T3_resin_uptake",
    "Total_serum_thyroxin",
    "Total_serum_triiodothyronine",
    "TSH",
    "Max_diff_TSH_after_TRH"
]
TARGET_NAME = "thyroid_class"
CLASS_NAMES = ["Normal", "Hyperthyroid", "Hypothyroid"]

# ─── Model Hyperparameters ────────────────────────────────────────────────────
# Gaussian NB
GNB_VAR_SMOOTHING_RANGE = [1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

# Logistic Regression (for ensemble)
LR_C_RANGE = [0.01, 0.1, 1.0, 10.0]
LR_MAX_ITER = 1000

# SVM (for ensemble)
SVM_C_RANGE = [0.1, 1.0, 10.0]
SVM_KERNEL = "rbf"

# Random Forest (for ensemble)
RF_N_ESTIMATORS = [50, 100, 200]
RF_MAX_DEPTH = [3, 5, 7, None]

# KNN (for ensemble)
KNN_N_NEIGHBORS = [3, 5, 7, 9]

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
