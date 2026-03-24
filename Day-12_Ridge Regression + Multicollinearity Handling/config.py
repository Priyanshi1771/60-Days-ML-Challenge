"""Day 12: Blood Pressure Prediction — Config"""
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
    "age", "weight_kg", "height_cm", "bmi", "waist_cm", "heart_rate",
    "cholesterol_total", "cholesterol_ldl", "cholesterol_hdl", "triglycerides",
    "fasting_glucose", "creatinine", "sodium_intake_mg", "potassium_intake_mg",
    "alcohol_drinks_week", "exercise_hours_week", "sleep_hours",
    "stress_level", "smoking", "diabetes", "family_htn"
]
TARGET_SYS = "systolic_bp"
TARGET_DIA = "diastolic_bp"

# Known collinear pairs for analysis
COLLINEAR_PAIRS = [
    ("weight_kg", "bmi"),
    ("cholesterol_total", "cholesterol_ldl"),
    ("sodium_intake_mg", "potassium_intake_mg"),
]

RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# GPU Neural Net
NN_HIDDEN = [128, 64, 32]
NN_EPOCHS = 120
NN_BATCH = 128
NN_LR = 1e-3

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
