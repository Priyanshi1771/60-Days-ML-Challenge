"""
Day 11: ICU Mortality Risk Prediction — Config
Dataset: MIMIC-IV inspired ICU patient data
Models: Linear/Polynomial Regression + GPU Neural Net Regressor
Focus: Polynomial features, regression metrics (MAE, RMSE, R²)
"""
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

FEATURE_NAMES = [
    "age", "heart_rate", "systolic_bp", "diastolic_bp", "resp_rate",
    "temperature", "spo2", "gcs_score", "bun", "creatinine",
    "sodium", "potassium", "hemoglobin", "wbc", "platelet",
    "lactate", "pao2_fio2", "urine_output_24h", "ventilator",
    "vasopressor", "prev_icu_stays", "los_before_icu"
]
TARGET_NAME = "mortality_risk"

# Polynomial
POLY_DEGREES = [1, 2, 3]

# GPU Neural Net regressor
NN_HIDDEN = [128, 64]
NN_EPOCHS = 100
NN_BATCH = 256
NN_LR = 1e-3

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
