"""
=============================================================================
 Day 7: Stroke Risk Prediction — Configuration
=============================================================================
 Dataset  : Kaggle Stroke Prediction (5110 samples, 11 features)
 Models   : XGBoost (primary), + LR, RF, SVM baselines
 Focus    : SHAP interpretability — understanding WHY the model predicts
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

# Features
NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
CATEGORICAL_FEATURES = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
TARGET_NAME = "stroke"
CLASS_NAMES = ["No Stroke", "Stroke"]

# XGBoost hyperparameters grid
XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [1, 10, 20],  # Handle 95:5 imbalance
}

# Reduced grid for faster search
XGB_PARAM_GRID_FAST = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "scale_pos_weight": [1, 15, 25],
}

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
