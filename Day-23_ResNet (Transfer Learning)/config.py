"""Day 23: Skin Lesion Classification — Config"""
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

IMG_SIZE = 224          # ResNet expects 224x224
NUM_CLASSES = 7
CLASS_NAMES = [
    "Melanocytic Nevi", "Melanoma", "Benign Keratosis",
    "Basal Cell Carcinoma", "Actinic Keratosis",
    "Vascular Lesion", "Dermatofibroma"
]
SYNTHETIC_N = 3500      # 500 per class

# Transfer Learning strategies to compare
STRATEGIES = ["scratch", "frozen", "finetune"]

# Training
EPOCHS = 25
BATCH_SIZE = 32
LR_SCRATCH = 1e-3
LR_FINETUNE = 1e-4     # lower LR for pretrained weights
LR_FROZEN = 1e-3       # only FC head trained
WEIGHT_DECAY = 1e-4
PATIENCE = 6

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
