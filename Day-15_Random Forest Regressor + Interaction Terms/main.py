"""
DAY 15: BMI Prediction
⚖️ Random Forest Regressor + Interaction Feature Engineering + GPU NN
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
for h in [logging.FileHandler(f"{config.LOG_DIR}/day15_experiment.log", mode='w'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    logger.addHandler(h)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, engineer_interactions, compare_with_without_interactions, preprocess_and_split
from model_training import train_random_forest, train_baselines, train_gpu_nn, save_models
from evaluation import (evaluate_all, plot_predictions, plot_bmi_categories,
                         plot_comparison, save_results)


def main():
    t0 = time.time()
    logging.info("╔" + "═" * 58 + "╗")
    logging.info("║  ⚖️ DAY 15: BMI PREDICTION                               ║")
    logging.info("║  🌲 Random Forest + Interaction Feature Engineering       ║")
    logging.info("║  ⚡ GPU Neural Net Comparison                            ║")
    logging.info("╚" + "═" * 58 + "╝")

    X_raw, y = load_data()
    explore_data(X_raw, y)

    # Feature engineering
    X_inter, inter_names, _ = engineer_interactions(X_raw, config.RAW_FEATURES)
    compare_with_without_interactions(X_raw, X_inter, y, config.RAW_FEATURES, inter_names)

    # Use interaction-augmented features
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(X_inter, y)

    rf_model, grid = train_random_forest(X_train, y_train)
    baselines = train_baselines(X_train, y_train)
    nn_model = train_gpu_nn(X_train, y_train)
    save_models(rf_model, baselines, nn_model, scaler)

    results_df, y_rf, y_nn = evaluate_all(rf_model, baselines, nn_model, X_test, y_test, inter_names)
    plot_predictions(y_test, y_rf, y_nn)
    plot_bmi_categories(y_test, y_rf)
    plot_comparison(results_df)
    save_results(results_df)

    logging.info(f"\n{'='*60}")
    logging.info(f"⚖️ DAY 15 COMPLETE | {time.time()-t0:.1f}s | Device: {config.DEVICE}")
    logging.info(f"  Best: {results_df.iloc[0]['Model']} | R²={results_df.iloc[0]['R²']:.4f}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
