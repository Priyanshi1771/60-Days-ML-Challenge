"""
DAY 11: ICU Mortality Risk Prediction
🏥 Phase 2 Begins — Regression & Time-Series
⚡ Polynomial Features + GPU Neural Net Regressor
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
fh = logging.FileHandler(f"{config.LOG_DIR}/day11_experiment.log", mode='w')
fh.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(ch)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, preprocess_and_split
from model_training import train_polynomial_models, train_gpu_neural_net, save_models
from evaluation import (evaluate_all, plot_predictions, plot_model_comparison,
                         plot_feature_importance, save_results)


def main():
    total_start = time.time()
    logging.info("╔" + "═" * 58 + "╗")
    logging.info("║  🏥 DAY 11: ICU MORTALITY RISK PREDICTION                ║")
    logging.info("║  📈 Polynomial Regression + GPU Neural Net               ║")
    logging.info("║  📊 Phase 2 Begins: Regression & Time-Series!            ║")
    logging.info("╚" + "═" * 58 + "╝")

    X, y = load_data()
    explore_data(X, y)
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(X, y)

    # Polynomial regression (degree 1, 2, 3)
    poly_results = train_polynomial_models(X_train, y_train)

    # GPU neural net regressor
    nn_model, nn_history = train_gpu_neural_net(X_train, y_train)

    save_models(poly_results, nn_model, scaler)

    # Evaluate all
    results_df = evaluate_all(poly_results, nn_model, X_test, y_test)
    plot_predictions(poly_results, nn_model, X_test, y_test)
    plot_model_comparison(results_df)
    plot_feature_importance(poly_results, config.FEATURE_NAMES)
    save_results(results_df)

    logging.info(f"\n{'='*60}")
    logging.info(f"🏥 DAY 11 COMPLETE | {time.time()-total_start:.1f}s | Device: {config.DEVICE}")
    logging.info(f"  Best: {results_df.iloc[0]['Model']} | R²={results_df.iloc[0]['R²']:.4f}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
