"""Day 13: COVID-19 Case Forecasting — Config"""
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

# Time-series config
N_DAYS = 730          # ~2 years of daily data
TRAIN_RATIO = 0.80
FORECAST_HORIZON = 30 # predict 30 days ahead

# ARIMA search ranges
ARIMA_P_RANGE = range(0, 4)
ARIMA_D_RANGE = range(0, 3)
ARIMA_Q_RANGE = range(0, 4)

# Exponential Smoothing
ETS_SEASONAL_PERIOD = 7  # weekly seasonality

# GPU LSTM
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_LOOKBACK = 21    # use 21 days to predict next day
LSTM_EPOCHS = 80
LSTM_BATCH = 32
LSTM_LR = 1e-3

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
