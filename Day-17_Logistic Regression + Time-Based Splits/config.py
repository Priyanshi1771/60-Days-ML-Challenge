"""Day 17: Hospital Readmission Risk — Config"""
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
N_PATIENTS = 8000
READMISSION_RATE = 0.18  # ~18% readmitted within 30 days (realistic)
TIME_SPAN_YEARS = 4       # 4 years of admission data

FEATURE_NAMES = [
    "age", "gender", "n_prev_admissions", "los_days", "n_diagnoses",
    "n_procedures", "n_medications", "n_lab_tests", "hba1c_result",
    "discharge_disposition", "admission_source", "primary_diag_group",
    "n_outpatient_visits", "n_er_visits", "n_inpatient_visits",
    "insulin_prescribed", "metformin_prescribed", "diabetic",
    "comorbidity_score", "payer_code"
]
TARGET_NAME = "readmitted_30d"
CLASS_NAMES = ["Not Readmitted", "Readmitted <30d"]

# Logistic Regression
LR_PARAM_GRID = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "solver": ["saga"],
    "class_weight": ["balanced", None],
}

# GPU NN
NN_HIDDEN = [128, 64]
NN_EPOCHS = 60
NN_BATCH = 256
NN_LR = 1e-3

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
