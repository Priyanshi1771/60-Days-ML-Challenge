"""Day 20: Viral Load Forecasting — Config"""
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

N_PATIENTS = 200
TIMEPOINTS = 52     # weekly for 1 year
LOOKBACK = 8

LSTM_HIDDEN = 64
LSTM_LAYERS = 2
GRU_HIDDEN = 64
GRU_LAYERS = 2
DROPOUT = 0.2
EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-3

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
