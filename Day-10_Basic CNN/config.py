"""
=============================================================================
 Day 10: Malaria Cell Classification — Configuration
=============================================================================
 Dataset  : Kaggle/NIH Malaria Cell Images (27,558 images, 2 classes)
 Model    : Custom CNN (built from scratch with PyTorch)
 Focus    : INTRODUCTION TO DEEP LEARNING — first CNN in the challenge
=============================================================================
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
IMG_SIZE = 64          # Resize to 64×64 (memory efficient, fast training)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
NUM_CLASSES = 2
CLASS_NAMES = ["Parasitized", "Uninfected"]
SYNTHETIC_N = 2000     # Fallback dataset size (1000 per class)

# CNN Architecture
CNN_CHANNELS = [3, 32, 64, 128]  # Input → Conv1 → Conv2 → Conv3
FC_HIDDEN = 256
DROPOUT = 0.4

# Training
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 3
EARLY_STOP_PATIENCE = 5
NUM_WORKERS = 0        # 0 for Windows compatibility; set 2-4 on Linux

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
