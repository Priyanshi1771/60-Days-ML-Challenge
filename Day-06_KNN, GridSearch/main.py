"""
=============================================================================
 DAY 6: Chronic Kidney Disease Prediction
=============================================================================
 🎯 OBJECTIVE: Predict CKD using KNN with exhaustive GridSearchCV tuning
 🧠 KEY LEARNING: GridSearch hyperparameter optimization deep-dive
 📊 MODELS: KNN (primary, tuned), LR, SVM, RF, DT (baselines)
 📁 DATASET: UCI Chronic Kidney Disease (400 samples, 24 features)
 
 🔧 ENGINEERING PRACTICES
    ✅ No data leakage (imputer + scaler fit on train only)
    ✅ Stratified splits (preserves CKD/Not-CKD ratio)
    ✅ 10-fold stratified CV for all model selection
    ✅ Multiple metrics (Accuracy, F1, Precision, Recall, AUC-ROC)
    ✅ Error analysis (FP vs FN — critical in medicine)
    ✅ Full reproducibility (fixed seeds)
    ✅ Modular code + comprehensive logging + model persistence
    
 ▶️ USAGE: cd day06_kidney_disease && python main.py
=============================================================================
"""

from evaluation import (
    evaluate_all, plot_confusion_matrices, plot_model_comparison,
    plot_roc_curves, error_analysis, save_results
)
from model_training import train_knn_gridsearch, train_baselines, save_models
from data_pipeline import load_data, explore_data, preprocess_and_split
import config
import matplotlib
import os
import sys
import time
import logging
import warnings
import numpy as np
warnings.filterwarnings("ignore")
matplotlib.use('Agg')


for d in [config.DATA_DIR, config.MODEL_DIR, config.PLOT_DIR, config.LOG_DIR, config.OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(f"{config.LOG_DIR}/day06_experiment.log", mode='w')
fh.setFormatter(logging.Formatter(
    config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter(
    config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(ch)

np.random.seed(config.RANDOM_SEED)


def main():
    total_start = time.time()

    logging.info("╔" + "═" * 58 + "╗")
    logging.info("║  🫘 DAY 6: CHRONIC KIDNEY DISEASE PREDICTION             ║")
    logging.info("║  🔬 Model: KNN + GridSearchCV Hyperparameter Tuning      ║")
    logging.info("║  📊 Dataset: UCI CKD (400 samples, 24 features)          ║")
    logging.info("╚" + "═" * 58 + "╝")

    # Phase 1: Data
    df = load_data()
    class_dist = explore_data(df)
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_cols = preprocess_and_split(
        df)

    # Phase 2: Training
    best_knn, grid, gridsearch_df = train_knn_gridsearch(X_train, y_train)
    baselines, baseline_scores = train_baselines(X_train, y_train)
    save_models(best_knn, baselines, scaler, label_encoders, grid)

    # Phase 3: Evaluation
    results_df, all_models = evaluate_all(
        best_knn, baselines, X_train, X_test, y_train, y_test, grid.best_score_)
    plot_confusion_matrices(all_models, X_test, y_test)
    plot_model_comparison(results_df)
    plot_roc_curves(all_models, X_test, y_test)
    error_analysis(best_knn, X_test, y_test, feature_cols)
    save_results(results_df)

    total_time = time.time() - total_start
    logging.info("\n" + "=" * 60)
    logging.info("🫘 DAY 6 COMPLETE")
    logging.info("=" * 60)
    logging.info(f"  Total runtime: {total_time:.2f}s")
    logging.info(f"  Best KNN params: {grid.best_params_}")
    logging.info(f"  Best model: {results_df.iloc[0]['Model']}")
    logging.info(f"  Best F1: {results_df.iloc[0]['F1 (Weighted)']:.4f}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
