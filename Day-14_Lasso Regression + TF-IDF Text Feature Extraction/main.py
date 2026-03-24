"""
DAY 14: Drug Response Prediction
💊 Lasso Regression + TF-IDF Text Feature Extraction + GPU Neural Net
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
for h in [logging.FileHandler(f"{config.LOG_DIR}/day14_experiment.log", mode='w'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    logger.addHandler(h)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, extract_features
from model_training import train_lasso_sweep, train_baselines, train_gpu_nn, save_models
from evaluation import (evaluate_all, plot_predictions, plot_comparison, save_results)


def main():
    t0 = time.time()
    logging.info("╔" + "═" * 58 + "╗")
    logging.info("║  💊 DAY 14: DRUG RESPONSE PREDICTION                     ║")
    logging.info("║  📝 TF-IDF Text Features + Lasso Feature Selection       ║")
    logging.info("║  ⚡ GPU Neural Net Comparison                            ║")
    logging.info("╚" + "═" * 58 + "╝")

    df = load_data()
    explore_data(df)
    X_train, X_test, y_train, y_test, tfidf, scaler = extract_features(df)
    del df  # free memory

    lasso_results = train_lasso_sweep(X_train, y_train)
    baselines = train_baselines(X_train, y_train)
    nn_model = train_gpu_nn(X_train, y_train)

    save_models(lasso_results, baselines, nn_model, tfidf, scaler)

    results_df = evaluate_all(lasso_results, baselines, nn_model, X_test, y_test, tfidf)
    plot_predictions(lasso_results, nn_model, X_test, y_test)
    plot_comparison(results_df)
    save_results(results_df)

    logging.info(f"\n{'='*60}")
    logging.info(f"💊 DAY 14 COMPLETE | {time.time()-t0:.1f}s | Device: {config.DEVICE}")
    logging.info(f"  Best: {results_df.iloc[0]['Model']} | RMSE={results_df.iloc[0]['RMSE']:.4f}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
