"""Day 15: BMI Prediction — Config"""
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

RAW_FEATURES = [
    "age", "gender", "height_cm", "weight_kg", "waist_cm", "hip_cm",
    "neck_cm", "chest_cm", "abdomen_cm", "thigh_cm", "knee_cm",
    "ankle_cm", "bicep_cm", "forearm_cm", "wrist_cm",
    "calories_daily", "protein_g", "carbs_g", "fat_g",
    "exercise_hrs_week", "sedentary_hrs_day", "sleep_hrs",
    "water_liters", "alcohol_weekly", "smoker"
]
TARGET_NAME = "bmi"

# Interaction pairs to engineer
INTERACTION_PAIRS = [
    ("waist_cm", "hip_cm"),          # waist-to-hip ratio
    ("weight_kg", "height_cm"),      # BMI-like ratio
    ("calories_daily", "exercise_hrs_week"),  # caloric balance
    ("sedentary_hrs_day", "exercise_hrs_week"),  # activity ratio
    ("fat_g", "protein_g"),          # macro ratio
    ("age", "exercise_hrs_week"),    # age-activity interaction
    ("waist_cm", "chest_cm"),       # body shape ratio
    ("abdomen_cm", "height_cm"),    # truncal obesity index
]

# RF grid
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 12, 16, None],
    "min_samples_leaf": [2, 5, 10],
}

# GPU NN
NN_HIDDEN = [128, 64]
NN_EPOCHS = 80
NN_BATCH = 128
NN_LR = 1e-3

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
