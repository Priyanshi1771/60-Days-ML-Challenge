"""
=============================================================================
 Day 8: Anemia Detection — Configuration
=============================================================================
 Dataset  : Kaggle Anemia Dataset (blood test results)
 Models   : AdaBoost (primary), + LR, RF, SVM, DT baselines
 Focus    : Feature scaling strategies + outlier removal techniques
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
N_SPLITS = 10

# Anemia dataset features (CBC — Complete Blood Count)
FEATURE_NAMES = [
    "Gender", "Hemoglobin", "MCH", "MCHC", "MCV",
    "RBC_count", "WBC_count", "Platelet_count"
]
TARGET_NAME = "Result"
CLASS_NAMES = ["Not Anemic", "Anemic"]

# AdaBoost hyperparameter grid
ADA_PARAM_GRID = {
    "n_estimators": [50, 100, 200, 300, 500],
    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
}

# Outlier removal methods to compare
OUTLIER_METHODS = ["none", "iqr", "zscore", "isolation_forest"]
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0

# Scaling methods to compare
SCALING_METHODS = ["none", "standard", "minmax", "robust"]

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
