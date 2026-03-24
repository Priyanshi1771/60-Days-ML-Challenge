"""
Day 1 - Heart Disease Prediction
Configuration file: All hyperparameters and paths in one place.

WHY THIS MATTERS:
-----------------
Hardcoding values like random seeds, file paths, or hyperparameters scattered across
multiple files is a recipe for debugging nightmares. A single config file means:
  1. You can reproduce any experiment by saving this file.
  2. Changing a hyperparameter doesn't require hunting through 5 files.
  3. Anyone reading your code immediately knows what's tunable.

COMMON MISTAKE: Using different random seeds in different files, leading to
non-reproducible results. Centralizing the seed here prevents that.
"""

import os

# ============================================================
# PATHS
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
PLOT_DIR = os.path.join(PROJECT_DIR, "plots")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# ============================================================
# REPRODUCIBILITY
# ============================================================
RANDOM_SEED = 42  # Used EVERYWHERE: splits, model init, numpy, etc.

# ============================================================
# DATA
# ============================================================
TEST_SIZE = 0.2           # 80/20 train/test split
VALIDATION_FOLDS = 5      # Stratified K-Fold for cross-validation

# ============================================================
# MODEL HYPERPARAMETERS (Logistic Regression)
# ============================================================
# C = inverse regularization strength. Smaller C = stronger regularization.
# Think of it as: high C trusts the data more, low C trusts the prior (simpler model) more.
LR_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # Grid search range
LR_MAX_ITER = 1000        # Max iterations for convergence
LR_SOLVER = "lbfgs"       # Best for small datasets, supports L2 penalty

# ============================================================
# RANDOM FOREST (comparison baseline)
# ============================================================
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None        # Let trees grow fully, then prune via CV

# ============================================================
# EVALUATION
# ============================================================
CLASSIFICATION_THRESHOLD = 0.5  # Default; we'll also explore threshold tuning
