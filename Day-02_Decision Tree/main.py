"""
Day 2 - Breast Cancer Diagnosis with Decision Trees
Main Script: Complete pipeline from data to multi-model comparison.

HOW TO RUN:
    cd day02_breast_cancer
    python main.py

WHAT THIS PIPELINE PRODUCES:
1. Exploratory data analysis with correlation heatmap
2. Feature distribution plots (malignant vs benign)
3. Three models: Unpruned → Pre-pruned → CCP-pruned (shows pruning impact)
4. Depth vs performance curve (bias-variance tradeoff visualization)
5. CCP pruning path (cost-complexity analysis)
6. Tree visualization (actual decision rules — the "explainable AI" of Day 2)
7. Side-by-side confusion matrices and ROC curves
8. Gini vs Permutation feature importance comparison
9. Detailed error analysis (focus on missed cancers)

KEY LEARNING FOR DAY 2:
  - Decision Trees are powerful but overfit without pruning
  - CCP (post-pruning) is more principled than pre-pruning
  - Trees don't need scaling (unlike Day 1's Logistic Regression)
  - Gini importance is biased — always cross-check with permutation importance
  - The bias-variance tradeoff visualized through tree depth is FUNDAMENTAL
"""

import numpy as np
import random
import os
import sys
import logging
import json
from datetime import datetime

# ============================================================
# REPRODUCIBILITY
# ============================================================
from config import RANDOM_SEED, LOG_DIR, PLOT_DIR, MODEL_DIR, OUTPUT_DIR

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Import modules
from data_pipeline import (
    load_data, explore_data, plot_correlation_matrix,
    plot_feature_distributions, preprocess_and_split
)
from model_training import (
    train_unpruned_tree, train_prepruned_tree, train_ccp_pruned_tree,
    visualize_tree, analyze_depth_performance, save_model
)
from evaluation import (
    evaluate_model, compare_models, plot_confusion_matrices,
    plot_roc_curves, analyze_feature_importance, error_analysis
)


def setup_logging():
    """Configure logging to console and file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"experiment_{timestamp}.log")

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


def main():
    """Execute the complete Day 2 pipeline."""
    log_file = setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("  DAY 2: BREAST CANCER DIAGNOSIS — DECISION TREES")
    logger.info("  Focus: Cross-validation, Pre-pruning vs CCP, Tree Visualization")
    logger.info("=" * 70)

    for dir_path in [PLOT_DIR, MODEL_DIR, OUTPUT_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # ===========================================================
    # STEP 1: LOAD & EXPLORE DATA
    # ===========================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: DATA LOADING & EXPLORATION")
    logger.info("=" * 60)

    df, feature_names = load_data()
    eda_results = explore_data(df)
    plot_correlation_matrix(df, feature_names)
    plot_feature_distributions(df, feature_names)

    # ===========================================================
    # STEP 2: PREPROCESS & SPLIT
    # ===========================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: PREPROCESSING & SPLITTING")
    logger.info("=" * 60)

    data = preprocess_and_split(df, feature_names)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    # ===========================================================
    # STEP 3: DEPTH vs PERFORMANCE ANALYSIS
    # ===========================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: BIAS-VARIANCE ANALYSIS (Depth vs Performance)")
    logger.info("=" * 60)

    optimal_depth = analyze_depth_performance(X_train, y_train)

    # ===========================================================
    # STEP 4: TRAIN ALL THREE MODELS
    # ===========================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: MODEL TRAINING — Three Pruning Strategies")
    logger.info("=" * 60)

    # Model 1: Unpruned (overfitting baseline)
    unpruned_tree = train_unpruned_tree(X_train, y_train, X_test, y_test)
    save_model(unpruned_tree, "dt_unpruned.pkl")

    # Model 2: Pre-pruned (GridSearchCV)
    prepruned_tree, grid_search = train_prepruned_tree(X_train, y_train)
    save_model(prepruned_tree, "dt_prepruned_best.pkl")

    # Model 3: CCP-pruned (cost-complexity)
    ccp_tree, best_alpha, ccp_data = train_ccp_pruned_tree(X_train, y_train)
    save_model(ccp_tree, "dt_ccp_pruned.pkl")

    # ===========================================================
    # STEP 5: VISUALIZE TREES
    # ===========================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: TREE VISUALIZATION")
    logger.info("=" * 60)

    visualize_tree(prepruned_tree, feature_names, "Pre-pruned Tree")
    visualize_tree(ccp_tree, feature_names, "CCP-pruned Tree")

    # ===========================================================
    # STEP 6: EVALUATE ALL MODELS
    # ===========================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: COMPREHENSIVE EVALUATION")
    logger.info("=" * 60)

    all_results = {}

    # Evaluate each model
    models = {
        'Unpruned': unpruned_tree,
        'Pre-pruned': prepruned_tree,
        'CCP-pruned': ccp_tree
    }

    for name, model in models.items():
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test, name)
        all_results[name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

    # Comparison visualizations
    compare_models(all_results)
    plot_confusion_matrices(all_results, y_test)
    plot_roc_curves(all_results, y_test)

    # ===========================================================
    # STEP 7: FEATURE IMPORTANCE (Best model)
    # ===========================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 60)

    # Use the CCP-pruned model (most principled approach)
    analyze_feature_importance(ccp_tree, X_test, y_test,
                               list(feature_names), "CCP-pruned Tree")

    # ===========================================================
    # STEP 8: ERROR ANALYSIS
    # ===========================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: ERROR ANALYSIS")
    logger.info("=" * 60)

    best_model_name = max(all_results, key=lambda k: all_results[k]['metrics']['f1'])
    error_df = error_analysis(
        models[best_model_name], X_test, y_test,
        all_results[best_model_name]['y_pred'],
        list(feature_names)
    )

    # ===========================================================
    # FINAL SUMMARY
    # ===========================================================
    logger.info("\n" + "=" * 70)
    logger.info("  FINAL SUMMARY — DAY 2")
    logger.info("=" * 70)

    logger.info(f"\n  Best model: {best_model_name}")
    best_metrics = all_results[best_model_name]['metrics']
    logger.info(f"  Accuracy:             {best_metrics['accuracy']:.4f}")
    logger.info(f"  Weighted F1:          {best_metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:              {best_metrics['auc_roc']:.4f}")
    logger.info(f"  Recall (malignant):   {best_metrics['recall_malignant']:.4f}")

    logger.info(f"\n  Tree complexity comparison:")
    for name, model in models.items():
        logger.info(f"    {name:>15}: depth={model.get_depth()}, leaves={model.get_n_leaves()}")

    # Save results
    results = {
        'experiment': 'Day 2 - Breast Cancer Diagnosis',
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'samples': eda_results['shape'][0],
            'features': eda_results['n_features'],
            'class_distribution': eda_results['class_distribution']
        },
        'models': {}
    }

    for name in all_results:
        results['models'][name] = {
            'metrics': all_results[name]['metrics'],
            'depth': models[name].get_depth(),
            'n_leaves': models[name].get_n_leaves()
        }

    results['best_model'] = best_model_name
    results['ccp_best_alpha'] = float(best_alpha)
    results['optimal_depth_by_cv'] = int(optimal_depth)

    results_path = os.path.join(OUTPUT_DIR, "experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")

    logger.info("\n" + "=" * 70)
    logger.info("  KEY TAKEAWAYS FROM DAY 2:")
    logger.info("  1. Unpruned trees memorize training data → always prune")
    logger.info("  2. CCP is more principled than ad-hoc pre-pruning")
    logger.info("  3. Trees don't need scaling (unlike LR from Day 1)")
    logger.info("  4. Gini importance ≠ true importance → use permutation")
    logger.info("  5. Tree visualization = built-in explainability")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
