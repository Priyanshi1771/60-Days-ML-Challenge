"""
Day 2 - Breast Cancer Diagnosis
Configuration file: All hyperparameters and paths in one place.

WHAT'S NEW vs Day 1:
---------------------
- 10-fold CV (up from 5) because dataset is larger (569 vs 303)
- Tree-specific hyperparameters: max_depth, min_samples, ccp_alpha
- Two splitting criteria to compare: Gini vs Entropy
- No scaler needed! Trees are scale-invariant.
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
RANDOM_SEED = 42

# ============================================================
# DATA
# ============================================================
TEST_SIZE = 0.2
VALIDATION_FOLDS = 10  # More folds = tighter CI, affordable with 569 samples

# ============================================================
# DECISION TREE HYPERPARAMETERS
# ============================================================
# --- Pre-pruning Grid Search ---
DT_PARAM_GRID = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
}

# --- Cost-Complexity Pruning (post-pruning) ---
# ccp_alpha range will be determined from the tree's cost-complexity path
# Higher alpha = more pruning = simpler tree
CCP_ALPHA_STEPS = 50  # Number of alpha values to test

# ============================================================
# EVALUATION
# ============================================================
CLASSIFICATION_THRESHOLD = 0.5
