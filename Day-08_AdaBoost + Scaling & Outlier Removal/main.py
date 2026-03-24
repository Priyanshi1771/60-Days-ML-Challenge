"""
=============================================================================
 DAY 8: Anemia Detection
=============================================================================
 🩸 OBJECTIVE: Detect anemia from Complete Blood Count (CBC) results
 🧪 KEY LEARNING: Scaling strategies + outlier removal techniques
 ⚡ MODEL: AdaBoost (Adaptive Boosting) — sequential weak learners
 📊 DATASET: Kaggle Anemia (1421 patients, 8 CBC features)
 
 ▶️ USAGE: cd day08_anemia_detection && python main.py
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
fh = logging.FileHandler(f"{config.LOG_DIR}/day08_experiment.log", mode='w')
fh.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(ch)
np.random.seed(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, preprocess_and_split
from model_training import run_ablation_study, train_adaboost, train_baselines, save_models
from evaluation import (evaluate_all, plot_confusion_matrices, plot_roc_curves,
                         plot_model_comparison, error_analysis, save_results)


def main():
    total_start = time.time()
    logging.info("╔" + "═" * 58 + "╗")
    logging.info("║  🩸 DAY 8: ANEMIA DETECTION                              ║")
    logging.info("║  🧪 Model: AdaBoost + Scaling & Outlier Removal          ║")
    logging.info("║  📊 Dataset: Kaggle Anemia (1421 patients, 8 CBC feats)  ║")
    logging.info("╚" + "═" * 58 + "╝")

    # Phase 1: Data
    df = load_data()
    explore_data(df)
    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    # Phase 2: Ablation Study (4 outlier × 4 scaling = 16 combos)
    ablation_df, best_outlier, best_scale = run_ablation_study(X_train, y_train)

    # Phase 3: Train AdaBoost with best preprocessing
    best_ada, grid, scaler, X_train_clean, y_train_clean, X_train_scaled = \
        train_adaboost(X_train, y_train, best_outlier, best_scale)

    # Phase 4: Train baselines (using same preprocessing)
    baselines, _ = train_baselines(X_train_scaled, y_train_clean)
    save_models(best_ada, baselines, scaler, grid)

    # Phase 5: Evaluate
    results_df, all_models, X_test_scaled = evaluate_all(
        best_ada, baselines, X_train, X_test, y_train, y_test, best_outlier, best_scale)
    plot_confusion_matrices(all_models, X_test_scaled, y_test)
    plot_roc_curves(all_models, X_test_scaled, y_test)
    plot_model_comparison(results_df)
    error_analysis(best_ada, X_test_scaled, y_test, config.FEATURE_NAMES)
    save_results(results_df, ablation_df)

    total_time = time.time() - total_start
    logging.info(f"\n{'='*60}")
    logging.info("🩸 DAY 8 COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"  Runtime: {total_time:.1f}s")
    logging.info(f"  Best preprocessing: {best_outlier} + {best_scale}")
    logging.info(f"  Best AdaBoost params: {grid.best_params_}")
    logging.info(f"  Best model: {results_df.iloc[0]['Model']} | F1={results_df.iloc[0]['F1 (Weighted)']:.4f}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
