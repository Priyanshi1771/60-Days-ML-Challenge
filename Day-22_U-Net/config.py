"""Day 22: Brain Tumor Segmentation — Config"""
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

IMG_SIZE = 128
SYNTHETIC_N = 2000  # MRI slices

# U-Net architecture
ENCODER_CH = [1, 32, 64, 128, 256]  # grayscale MRI
BOTTLENECK_CH = 512
DECODER_CH = [256, 128, 64, 32]

# Training
EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
