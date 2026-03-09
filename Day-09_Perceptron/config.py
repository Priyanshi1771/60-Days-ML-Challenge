"""
=============================================================================
 Day 9: Hepatitis Diagnosis — Configuration
=============================================================================
 Dataset  : UCI Hepatitis (155 samples, 19 features) — SMALL dataset
 Models   : Perceptron (primary), + LR, SVM, RF, MLP baselines
 Focus    : ROC curve analysis deep-dive + threshold optimization
=============================================================================
"""
import matplotlib
matplotlib.use('Agg')
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_SPLITS = 5  # Small dataset → 5-fold (not 10)

# UCI Hepatitis features
FEATURE_NAMES = [
    "Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malaise",
    "Anorexia", "Liver_Big", "Liver_Firm", "Spleen_Palpable",
    "Spiders", "Ascites", "Varices", "Bilirubin", "Alk_Phosphate",
    "SGOT", "Albumin", "Protime", "Histology"
]
NUMERIC_FEATURES = ["Age", "Bilirubin", "Alk_Phosphate", "SGOT", "Albumin", "Protime"]
BINARY_FEATURES = ["Sex", "Steroid", "Antivirals", "Fatigue", "Malaise",
                    "Anorexia", "Liver_Big", "Liver_Firm", "Spleen_Palpable",
                    "Spiders", "Ascites", "Varices", "Histology"]
TARGET_NAME = "Class"
CLASS_NAMES = ["Die", "Live"]

# Perceptron grid
PERCEPTRON_PARAM_GRID = {
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "max_iter": [500, 1000, 2000],
    "penalty": ["l2", "l1", "elasticnet"],
    "eta0": [0.1, 0.5, 1.0],
}

# ROC threshold analysis
THRESHOLD_RANGE = (0.05, 0.95, 50)  # start, stop, num_points

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
