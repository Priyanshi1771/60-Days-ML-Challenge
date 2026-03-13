"""
DAY 13: COVID-19 Case Forecasting
🦠 ARIMA + Exponential Smoothing + GPU LSTM
📈 First time-series project in the 60-day challenge!
"""
import matplotlib
matplotlib.use('Agg')
import os, sys, time, logging, warnings, numpy as np, torch
warnings.filterwarnings("ignore")

import config
for d in [config.DATA_DIR, config.MODEL_DIR, config.PLOT_DIR, config.LOG_DIR, config.OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in [logging.FileHandler(f"{config.LOG_DIR}/day13_experiment.log", mode='w'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    logger.addHandler(h)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, split_data, prepare_lstm_data
from model_training import train_arima, train_exponential_smoothing, train_gpu_lstm, save_models
from evaluation import (evaluate_all, plot_forecasts, plot_model_comparison,
                         plot_zoom_forecast, save_results)


def main():
    t0 = time.time()
    logging.info("╔" + "═" * 58 + "╗")
    logging.info("║  🦠 DAY 13: COVID-19 CASE FORECASTING                    ║")
    logging.info("║  📈 ARIMA + Exponential Smoothing + GPU LSTM             ║")
    logging.info("║  🕐 First Time-Series Project!                           ║")
    logging.info("╚" + "═" * 58 + "╝")

    # Data
    df = load_data()
    explore_data(df)
    train_df, test_df = split_data(df)

    # Classical: ARIMA
    arima_model, arima_fc = train_arima(train_df, test_df)

    # Classical: Exponential Smoothing
    ets_model, ets_fc = train_exponential_smoothing(train_df, test_df)

    # Deep Learning: GPU LSTM
    X_tr, y_tr, X_te, y_te, scaler = prepare_lstm_data(train_df["cases"], test_df["cases"])
    lstm_model, lstm_fc = train_gpu_lstm(X_tr, y_tr, X_te, y_te, scaler)

    save_models(arima_model, ets_model, lstm_model)

    # Evaluate
    results_df = evaluate_all(test_df, arima_fc, ets_fc, lstm_fc)
    plot_forecasts(train_df, test_df, arima_fc, ets_fc, lstm_fc)
    plot_model_comparison(results_df)
    plot_zoom_forecast(test_df, arima_fc, ets_fc, lstm_fc)
    save_results(results_df)

    logging.info(f"\n{'='*60}")
    logging.info(f"🦠 DAY 13 COMPLETE | {time.time()-t0:.1f}s | Device: {config.DEVICE}")
    logging.info(f"  Best: {results_df.iloc[0]['Model']} | RMSE={results_df.iloc[0]['RMSE']:.1f}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
