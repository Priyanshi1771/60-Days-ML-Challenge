"""
Day 1 - Heart Disease Prediction
Main Script: Orchestrates the complete ML pipeline from data to evaluation.

HOW TO RUN:
-----------
    cd day01_heart_disease
    python main.py

WHAT HAPPENS WHEN YOU RUN THIS:
---------------------------------
1. Sets up reproducibility (seeds) and logging
2. Loads and explores the heart disease dataset
3. Preprocesses data (scaling, splitting) — NO LEAKAGE
4. Trains Logistic Regression with GridSearchCV
5. Trains Random Forest as a comparison baseline
6. Evaluates both models with comprehensive metrics
7. Generates all plots (confusion matrix, ROC, feature importance, threshold)
8. Performs error analysis
9. Saves trained models to disk
10. Prints a final comparison and recommendations

FOLDER STRUCTURE:
-----------------
day01_heart_disease/
├── main.py              ← YOU ARE HERE (run this)
├── config.py            ← All hyperparameters and paths
├── data_pipeline.py     ← Data loading, preprocessing, splitting
├── model_training.py    ← Model training and persistence
├── evaluation.py        ← Metrics, plots, error analysis
├── data/                ← Raw data storage
├── models/              ← Saved trained models
├── logs/                ← Training logs
├── plots/               ← Generated visualizations
└── outputs/             ← Final results and reports

WHY THIS STRUCTURE?
--------------------
Modular code is:
  1. TESTABLE: You can test data_pipeline.py independently
  2. REUSABLE: swap models without touching data code
  3. DEBUGGABLE: bugs are isolated to one module
  4. READABLE: new team members can navigate easily

COMMON BEGINNER MISTAKE: One giant 500-line notebook with everything mixed together.
That works for exploration but is UNMAINTAINABLE for anything serious.
"""

import numpy as np
import random
import os
import sys
import logging
import json
from datetime import datetime

# ============================================================
# 0. REPRODUCIBILITY SETUP (Must be FIRST!)
# ============================================================
"""
WHY REPRODUCIBILITY MATTERS:
If you run the same code twice and get different results, how do you know
if a change you made helped or if you just got lucky?

Setting seeds ensures:
  - Same random splits every time
  - Same model initialization
  - Same shuffling order
  - Deterministic results for debugging

COMMON MISTAKE: Setting the seed in only one place. You need to seed:
  1. Python's built-in random
  2. NumPy's random
  3. The ML library (sklearn uses numpy's seed)
  4. For PyTorch (later days): torch.manual_seed() AND torch.cuda.manual_seed_all()
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from config import RANDOM_SEED, LOG_DIR, PLOT_DIR, MODEL_DIR, OUTPUT_DIR


# Set ALL random seeds
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Import our modules
from data_pipeline import load_data, explore_data, preprocess_and_split
from model_training import train_logistic_regression, train_random_forest, save_model
from evaluation import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    analyze_feature_importance, find_optimal_threshold, error_analysis
)


# ============================================================
# 1. LOGGING SETUP
# ============================================================
def setup_logging():
    """
    Configure logging to both console and file.

    WHY LOGGING (not just print statements)?
    ------------------------------------------
    1. Log files persist after the script ends — you can review later
    2. Logging levels (DEBUG, INFO, WARNING, ERROR) let you filter noise
    3. Timestamps help you track how long each step takes
    4. In production, logs are essential for monitoring and debugging

    COMMON MISTAKE: Using print() everywhere. It works for small scripts,
    but you can't search, filter, or redirect print output easily.
    After Day 1, you should NEVER use print() in serious ML code.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"experiment_{timestamp}.log")

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler (shows INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler (saves everything including DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


# ============================================================
# 2. MAIN PIPELINE
# ============================================================
def main():
    """
    Execute the complete Heart Disease Prediction pipeline.

    This is the main orchestrator — it calls all modules in the correct order,
    handles errors gracefully, and produces a final summary.
    """
    log_file = setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("  DAY 1: HEART DISEASE PREDICTION - COMPLETE PIPELINE")
    logger.info("  Model: Logistic Regression | Dataset: UCI Heart Disease")
    logger.info("=" * 70)

    # Ensure output directories exist
    for dir_path in [PLOT_DIR, MODEL_DIR, OUTPUT_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # ----------------------------------------------------------
    # STEP 1: Load and Explore Data
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: DATA LOADING AND EXPLORATION")
    logger.info("=" * 60)

    df = load_data()
    eda_results = explore_data(df)

    # ----------------------------------------------------------
    # STEP 2: Preprocess and Split
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: PREPROCESSING AND SPLITTING")
    logger.info("=" * 60)

    data = preprocess_and_split(df)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    X_train_raw = data['X_train_raw']
    X_test_raw = data['X_test_raw']

    # ----------------------------------------------------------
    # STEP 3: Train Logistic Regression
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("=" * 60)

    lr_model, lr_grid = train_logistic_regression(X_train, y_train)
    save_model(lr_model, "logistic_regression_best.pkl")
    save_model(data['scaler'], "standard_scaler.pkl")  # Save scaler too!

    # ----------------------------------------------------------
    # STEP 4: Train Random Forest (Comparison)
    # ----------------------------------------------------------
    rf_model, rf_cv_scores = train_random_forest(X_train_raw, y_train)
    save_model(rf_model, "random_forest_baseline.pkl")

    # ----------------------------------------------------------
    # STEP 5: Evaluate Both Models
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: EVALUATION")
    logger.info("=" * 60)

    # Logistic Regression evaluation
    lr_metrics, lr_pred, lr_proba = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression")
    plot_roc_curve(y_test, lr_proba, "Logistic Regression")
    analyze_feature_importance(lr_model, X_test, y_test, feature_names, "Logistic Regression")

    # Random Forest evaluation (uses unscaled data)
    rf_metrics, rf_pred, rf_proba = evaluate_model(rf_model, X_test_raw, y_test, "Random Forest")
    plot_confusion_matrix(y_test, rf_pred, "Random Forest")
    plot_roc_curve(y_test, rf_proba, "Random Forest")

    # ----------------------------------------------------------
    # STEP 6: Threshold Tuning
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: THRESHOLD TUNING")
    logger.info("=" * 60)

    thresholds = find_optimal_threshold(y_test, lr_proba, "Logistic Regression")

    # ----------------------------------------------------------
    # STEP 7: Error Analysis
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: ERROR ANALYSIS")
    logger.info("=" * 60)

    error_df = error_analysis(lr_model, X_test, y_test, lr_pred, lr_proba, feature_names)

    # ----------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("  FINAL SUMMARY")
    logger.info("=" * 70)

    logger.info("\n┌─────────────────────────────────────────────────────────┐")
    logger.info("│           MODEL COMPARISON                              │")
    logger.info("├─────────────────────┬──────────────┬──────────────┤")
    logger.info("│ Metric              │ Log. Regr.   │ Random Forest│")
    logger.info("├─────────────────────┼──────────────┼──────────────┤")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        lr_val = lr_metrics[metric]
        rf_val = rf_metrics[metric]
        winner = "←" if lr_val >= rf_val else "→"
        logger.info(f"│ {metric:>19} │   {lr_val:.4f}     │   {rf_val:.4f}     │ {winner}")
    logger.info("└─────────────────────┴──────────────┴──────────────┘")

    logger.info(f"\nOptimal Thresholds for Logistic Regression:")
    logger.info(f"  Best F1 threshold:    {thresholds['best_f1_threshold']:.3f}")
    logger.info(f"  90% Recall threshold: {thresholds['threshold_90_recall']:.3f}")
    logger.info(f"  Youden's J threshold: {thresholds['youden_threshold']:.3f}")

    # ----------------------------------------------------------
    # SAVE EXPERIMENT RESULTS
    # ----------------------------------------------------------
    results = {
        'experiment': 'Day 1 - Heart Disease Prediction',
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'samples': eda_results['shape'][0],
            'features': eda_results['shape'][1] - 1,
            'class_balance': eda_results['class_balance']
        },
        'logistic_regression': {
            'metrics': lr_metrics,
            'best_C': lr_grid.best_params_['C'],
            'cv_f1': lr_grid.best_score_
        },
        'random_forest': {
            'metrics': rf_metrics,
            'cv_f1_mean': float(rf_cv_scores.mean()),
            'cv_f1_std': float(rf_cv_scores.std())
        },
        'threshold_analysis': thresholds
    }

    results_path = os.path.join(OUTPUT_DIR, "experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nExperiment results saved to: {results_path}")

    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE!")
    logger.info(f"  Plots saved in: {PLOT_DIR}")
    logger.info(f"  Models saved in: {MODEL_DIR}")
    logger.info(f"  Logs saved in: {LOG_DIR}")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
