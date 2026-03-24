"""Day 19: Radiosensitivity Prediction — Config"""
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

DATASET_A_SAMPLES = 3000
DATASET_B_SAMPLES = 1500
N_FEATURES = 30

FEATURE_NAMES = [
    "TP53_mut", "BRCA1_expr", "BRCA2_expr", "ATM_expr", "RAD51_expr",
    "PARP1_expr", "MLH1_expr", "XRCC1_expr", "ERCC1_expr", "MGMT_expr",
    "tumor_size", "tumor_grade", "stage", "age", "ki67_pct",
    "hypoxia_score", "immune_score", "pdl1_expr", "tmb_mut_mb",
    "msi_score", "ploidy", "genome_instab", "apoptosis_score",
    "prolif_score", "dna_repair", "oxidative_stress",
    "angiogenesis", "inflammation", "metabolic", "stroma_score"
]

ELASTIC_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
ELASTIC_ALPHAS = [0.001, 0.01, 0.1, 1.0]

NN_HIDDEN = [128, 64]
NN_EPOCHS = 80
NN_BATCH = 128
NN_LR = 1e-3

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
