"""
=============================================================================
 DAY 7: Stroke Risk Prediction — XGBoost + SHAP Interpretability
=============================================================================
 🧠 OBJECTIVE: Predict stroke from patient demographics + clinical features
 ⚡ KEY LEARNING: SHAP — understanding WHY the model predicts what it does
 📊 DATASET: Kaggle Stroke (5110 samples, ~5% stroke rate)
 🎯 CHALLENGE: Extreme imbalance (95:5) — accuracy is meaningless here
 
 ▶️ USAGE: cd day07_stroke_prediction && python main.py
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
fh = logging.FileHandler(f"{config.LOG_DIR}/day07_experiment.log", mode='w')
fh.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(ch)
np.random.seed(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, preprocess_and_split
from model_training import train_xgboost, compute_shap_explanations, train_baselines, save_models
from evaluation import (evaluate_all, plot_confusion_matrices, plot_roc_and_pr_curves,
                         plot_model_comparison, error_analysis, save_results)


def main():
    total_start = time.time()
    logging.info("╔" + "═" * 58 + "╗")
    logging.info("║  🧠 DAY 7: STROKE RISK PREDICTION                       ║")
    logging.info("║  ⚡ Model: XGBoost + SHAP Interpretability               ║")
    logging.info("║  📊 Dataset: Kaggle Stroke (5110 samples, ~5% positive)  ║")
    logging.info("╚" + "═" * 58 + "╝")

    # Phase 1: Data
    df = load_data()
    explore_data(df)
    X_train_s, X_test_s, X_train, X_test, y_train, y_test, scaler, le, feat_cols = preprocess_and_split(df)

    # Phase 2: Train XGBoost
    best_xgb, grid = train_xgboost(X_train_s, y_train, X_train)

    # Phase 3: SHAP / Feature Importance
    compute_shap_explanations(best_xgb, X_train_s, X_test_s, feat_cols, y_test)

    # Phase 4: Baselines
    baselines, baseline_scores = train_baselines(X_train_s, y_train)
    save_models(best_xgb, baselines, scaler, le, grid)

    # Phase 5: Evaluation
    results_df, all_models = evaluate_all(best_xgb, baselines, X_train_s, X_test_s, y_train, y_test, grid.best_score_)
    plot_confusion_matrices(all_models, X_test_s, y_test)
    plot_roc_and_pr_curves(all_models, X_test_s, y_test)
    plot_model_comparison(results_df)
    error_analysis(best_xgb, X_test_s, y_test, feat_cols)
    save_results(results_df)

    total_time = time.time() - total_start
    logging.info(f"\n{'='*60}")
    logging.info("🧠 DAY 7 COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"  Runtime: {total_time:.1f}s")
    logging.info(f"  Best: {results_df.iloc[0]['Model']} | F1={results_df.iloc[0]['F1']:.4f}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
