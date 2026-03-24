"""Day 18: Gene Expression Prediction — Config"""
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
TEST_SIZE = 0.15
VAL_SIZE = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# High-dimensional genomic data
N_SAMPLES = 4000
N_GENES_INPUT = 500       # input gene probes (high-dimensional!)
N_GENES_TARGET = 1        # predict expression of 1 target gene
NOISE_GENES = 350         # ~70% of input features are noise (realistic for genomics)

# DL Architectures to compare
ARCHITECTURES = {
    "Shallow (1 layer)":   [128],
    "Medium (2 layers)":   [256, 128],
    "Deep (3 layers)":     [512, 256, 128],
    "Wide (2 layers)":     [1024, 512],
    "Bottleneck":          [256, 32, 128],
}

# Training
EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
PATIENCE = 10

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
