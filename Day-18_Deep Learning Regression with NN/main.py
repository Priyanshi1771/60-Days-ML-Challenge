"""
DAY 18: Gene Expression Prediction
DNA Deep Learning Regression -- 5 Architectures Compared on GPU
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
for h in [logging.FileHandler(f"{config.LOG_DIR}/day18_experiment.log", mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    logger.addHandler(h)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, preprocess_and_split, to_tensors
from model_training import train_all_architectures, train_sklearn_baselines, save_models
from evaluation import (evaluate_all, plot_best_predictions, plot_dl_vs_ml,
                         plot_capacity_vs_performance, save_results)


def main():
    t0 = time.time()
    logging.info("=" * 60)
    logging.info("  DAY 18: GENE EXPRESSION PREDICTION")
    logging.info("  DL Regression -- 5 GPU Architectures Compared")
    logging.info("  Dataset: 500 genes (150 signal + 350 noise)")
    logging.info("=" * 60)

    # Data
    X, y, is_signal, gene_names = load_data()
    explore_data(X, y, is_signal)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_and_split(X, y)

    # GPU tensors
    X_train_t, y_train_t = to_tensors(X_train, y_train)
    X_val_t, y_val_t = to_tensors(X_val, y_val)
    X_test_t, y_test_t = to_tensors(X_test, y_test)

    # DL: 5 architectures
    arch_results = train_all_architectures(X_train_t, y_train_t, X_val_t, y_val_t)

    # ML baselines
    baselines = train_sklearn_baselines(X_train, y_train)
    save_models(arch_results, baselines, scaler)

    # Evaluate
    results_df = evaluate_all(arch_results, baselines, X_test, y_test, X_test_t, y_test_t)
    plot_best_predictions(arch_results, baselines, X_test, y_test, X_test_t)
    plot_dl_vs_ml(results_df)
    plot_capacity_vs_performance(arch_results)
    save_results(results_df, arch_results)

    elapsed = time.time() - t0
    best = results_df.iloc[0]
    logging.info(f"\n{'='*60}")
    logging.info(f"  DAY 18 COMPLETE | {elapsed:.1f}s | Device: {config.DEVICE}")
    logging.info(f"  Best: {best['Model']} | RMSE={best['RMSE']:.4f} | R2={best['R2']:.4f}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
