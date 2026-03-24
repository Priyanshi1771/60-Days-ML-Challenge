"""
=============================================================================
 DAY 9: Hepatitis Diagnosis
=============================================================================
 🦠 OBJECTIVE: Predict hepatitis patient survival (Die vs Live)
 📈 KEY LEARNING: ROC curve analysis — thresholds, AUC, Youden's J, bootstrap CI
 ⚡ MODEL: Perceptron (simplest neural network) + CalibratedClassifierCV
 📊 DATASET: UCI Hepatitis (155 samples, 19 features, ~20% mortality)
 
 ▶️ USAGE: cd day09_hepatitis_diagnosis && python main.py
=============================================================================
"""
import matplotlib
matplotlib.use('Agg')
import os, sys, time, logging, warnings, numpy as np
warnings.filterwarnings("ignore")

import config
for d in [config.DATA_DIR, config.MODEL_DIR, config.PLOT_DIR, config.LOG_DIR, config.OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(f"{config.LOG_DIR}/day09_experiment.log", mode='w')
fh.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(ch)
np.random.seed(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, preprocess_and_split
from model_training import train_perceptron, train_baselines, save_models
from evaluation import (evaluate_all, plot_roc_deep_analysis, plot_confusion_matrices,
                         plot_model_comparison, error_analysis, save_results)


def main():
    total_start = time.time()
    logging.info("╔" + "═" * 58 + "╗")
    logging.info("║  🦠 DAY 9: HEPATITIS DIAGNOSIS                          ║")
    logging.info("║  📈 Model: Perceptron + ROC Curve Deep Analysis          ║")
    logging.info("║  📊 Dataset: UCI Hepatitis (155 samples, 19 features)    ║")
    logging.info("╚" + "═" * 58 + "╝")

    # Phase 1: Data
    X, y = load_data()
    explore_data(X, y)
    X_train, X_test, y_train, y_test, scaler, imputer = preprocess_and_split(X, y)

    # Phase 2: Perceptron + calibration
    perceptron, calibrated, grid = train_perceptron(X_train, y_train)

    # Phase 3: Baselines
    baselines = train_baselines(X_train, y_train)
    save_models(perceptron, calibrated, baselines, scaler, imputer, grid)

    # Phase 4: Evaluate
    results_df, all_models = evaluate_all(calibrated, baselines, X_train, X_test, y_train, y_test)
    plot_roc_deep_analysis(all_models, X_test, y_test)
    plot_confusion_matrices(all_models, X_test, y_test)
    plot_model_comparison(results_df)
    error_analysis(calibrated, X_test, y_test)
    save_results(results_df)

    total_time = time.time() - total_start
    logging.info(f"\n{'='*60}")
    logging.info("🦠 DAY 9 COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"  Runtime: {total_time:.1f}s")
    logging.info(f"  Best: {results_df.iloc[0]['Model']} | AUC={results_df.iloc[0]['AUC-ROC']:.4f}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
