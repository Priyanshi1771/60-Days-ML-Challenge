"""
DAY 17: Hospital Readmission Risk
🏥 Logistic Regression + Time-Based Splits + GPU Neural Net
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
for h in [logging.FileHandler(f"{config.LOG_DIR}/day17_experiment.log", mode='w'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    logger.addHandler(h)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, time_based_split, prepare_features
from model_training import (train_logistic_regression, train_baselines,
                             compare_temporal_vs_random, train_gpu_nn, save_models)
from evaluation import (evaluate_all, plot_roc_and_pr, plot_calibration,
                         plot_confusion_matrices, plot_comparison, save_results)


def main():
    t0 = time.time()
    logging.info("=" * 60)
    logging.info("  DAY 17: HOSPITAL READMISSION RISK")
    logging.info("  Time-Based Splits + Logistic Regression")
    logging.info("  GPU Neural Net Comparison")
    logging.info("=" * 60)

    df = load_data()
    explore_data(df)

    # KEY EXPERIMENT: temporal vs random split
    split_comparison = compare_temporal_vs_random(df, prepare_features)

    # Proper temporal split
    train_df, test_df = time_based_split(df)
    X_train, X_test, y_train, y_test, scaler = prepare_features(train_df, test_df)

    # Models
    lr_model, grid = train_logistic_regression(X_train, y_train)
    baselines = train_baselines(X_train, y_train)
    nn_model = train_gpu_nn(X_train, y_train)
    save_models(lr_model, baselines, nn_model, scaler)

    # Evaluate
    results_df, proba_dict, all_models = evaluate_all(lr_model, baselines, nn_model, X_test, y_test)
    plot_roc_and_pr(proba_dict, y_test)
    plot_calibration(proba_dict, y_test)
    plot_confusion_matrices(all_models, nn_model, X_test, y_test, proba_dict)
    plot_comparison(results_df)
    save_results(results_df, split_comparison)

    logging.info(f"\n{'='*60}")
    logging.info(f"🏥 DAY 17 COMPLETE | {time.time()-t0:.1f}s | Device: {config.DEVICE}")
    logging.info(f"  Best: {results_df.iloc[0]['Model']} | AUC={results_df.iloc[0]['AUC-ROC']:.4f}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
