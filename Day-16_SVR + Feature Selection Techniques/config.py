"""Day 16: Telomere Length Prediction — Config"""
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

# 40 genomic/clinical features — many are noise (feature selection challenge)
FEATURE_NAMES = [
    "age", "sex", "bmi", "waist_hip_ratio", "systolic_bp", "diastolic_bp",
    "resting_hr", "vo2_max", "sleep_hrs", "stress_score",
    "smoking_pack_years", "alcohol_units_wk", "exercise_hrs_wk",
    "wbc_count", "lymphocyte_pct", "neutrophil_pct", "crp_mg_l",
    "il6_pg_ml", "tnf_alpha", "oxidative_stress_score",
    "vitamin_d_ng", "folate_ng", "b12_pg", "homocysteine_umol",
    "hdl_chol", "ldl_chol", "triglycerides", "fasting_glucose", "hba1c",
    "cortisol_ug", "dhea_ug", "igf1_ng", "telomerase_activity",
    # Noise features (deliberately uninformative)
    "shoe_size", "eye_color_enc", "birth_month", "zip_first_digit",
    "favorite_number", "random_noise_1", "random_noise_2"
]
TARGET_NAME = "telomere_length_kb"  # kilobases

# Feature selection methods to compare
SELECTION_METHODS = ["none", "variance", "correlation", "mutual_info", "rfecv", "lasso"]
TOP_K_FEATURES = 15  # select top K from 40

# SVR hyperparameters
SVR_PARAM_GRID = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "epsilon": [0.01, 0.05, 0.1],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto"],
}

# GPU NN
NN_HIDDEN = [64, 32]
NN_EPOCHS = 80
NN_BATCH = 128
NN_LR = 1e-3

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
