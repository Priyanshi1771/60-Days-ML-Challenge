"""
=============================================================================
 DAY 5: Thyroid Disease Classification
=============================================================================
 
 🎯 OBJECTIVE
    Classify thyroid function into Normal, Hyperthyroid, or Hypothyroid
    using the UCI New Thyroid dataset.
 
 🧠 KEY LEARNING: Ensemble Voting Classifiers
    - Hard voting: majority vote among classifiers
    - Soft voting: average predicted probabilities
    - Weighted soft voting: weight classifiers by CV performance
    
 📊 MODELS
    Baseline: Gaussian Naive Bayes
    Individual: Logistic Regression, SVM, Random Forest, KNN
    Ensemble: Hard Voting, Soft Voting, Weighted Soft Voting
 
 📁 DATASET
    UCI New Thyroid (215 samples, 5 features, 3 classes)
    Features: T3 resin uptake, Total serum thyroxin, Total serum
              triiodothyronine, TSH, Max diff TSH after TRH injection
 
 🔧 ENGINEERING PRACTICES
    ✅ No data leakage (scaler fit on train only)
    ✅ Stratified splits (class balance preserved)
    ✅ 10-fold stratified CV for all model selection
    ✅ Multiple metrics (Accuracy, F1-weighted, F1-macro, Precision, Recall)
    ✅ Error analysis on misclassifications
    ✅ Full reproducibility (fixed seeds)
    ✅ Modular code (separate config, data, training, evaluation)
    ✅ Comprehensive logging
    ✅ All models + scaler saved together
    
 ▶️ USAGE
    cd day05_thyroid_disease
    python main.py
    
=============================================================================
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-GUI backend
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ─── Setup ────────────────────────────────────────────────────────────────────
import config

# Ensure directories exist
for d in [config.DATA_DIR, config.MODEL_DIR, config.PLOT_DIR, config.LOG_DIR, config.OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Logging Setup ────────────────────────────────────────────────────────────
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler (persistent log)
fh = logging.FileHandler(f"{config.LOG_DIR}/day05_experiment.log", mode='w')
fh.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(ch)

# ─── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(config.RANDOM_SEED)

# ─── Import Pipeline Modules ─────────────────────────────────────────────────
from data_pipeline import load_data, explore_data, preprocess_and_split
from model_training import (
    train_gaussian_nb, train_individual_classifiers,
    train_voting_ensembles, save_models
)
from evaluation import (
    evaluate_all_models, plot_confusion_matrices,
    plot_model_comparison, plot_ensemble_advantage,
    voting_agreement_analysis, error_analysis,
    save_results_report
)


def main():
    """Execute the full Day 5 pipeline."""
    
    total_start = time.time()
    
    logging.info("╔" + "═" * 58 + "╗")
    logging.info("║   DAY 5: THYROID DISEASE CLASSIFICATION                  ║")
    logging.info("║   Model: Naive Bayes + Ensemble Voting Classifiers       ║")
    logging.info("║   Dataset: UCI New Thyroid                               ║")
    logging.info("╚" + "═" * 58 + "╝")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: DATA LOADING & EDA
    # ═══════════════════════════════════════════════════════════════════════
    df = load_data()
    class_dist = explore_data(df)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: PREPROCESSING & SPLITTING
    # ═══════════════════════════════════════════════════════════════════════
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_and_split(df)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: MODEL TRAINING
    # ═══════════════════════════════════════════════════════════════════════
    
    # 3a. Gaussian Naive Bayes (baseline)
    gnb, gnb_cv = train_gaussian_nb(X_train, y_train)
    
    # 3b. Individual classifiers for ensemble
    classifiers, individual_cv = train_individual_classifiers(X_train, y_train)
    
    # 3c. Voting ensembles (hard, soft, weighted)
    ensembles, ensemble_cv, weights = train_voting_ensembles(X_train, y_train, classifiers)
    
    # 3d. Save all models
    save_models(gnb, classifiers, ensembles, scaler, label_encoder, weights)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    
    # 4a. Evaluate all models on test set
    results_df, all_models = evaluate_all_models(
        gnb, classifiers, ensembles, X_test, y_test,
        gnb_cv, individual_cv, ensemble_cv
    )
    
    # 4b. Confusion matrices
    plot_confusion_matrices(all_models, X_test, y_test)
    
    # 4c. Model comparison charts
    plot_model_comparison(results_df)
    
    # 4d. Ensemble advantage visualization
    plot_ensemble_advantage(results_df)
    
    # 4e. Voting agreement analysis
    voting_agreement_analysis(classifiers, ensembles, X_test, y_test)
    
    # 4f. Error analysis on best model
    best_model_name = results_df.iloc[0]["Model"]
    best_model = all_models[best_model_name]
    error_analysis(best_model, best_model_name, X_test, y_test, config.FEATURE_NAMES)
    
    # 4g. Save results
    save_results_report(results_df)
    
    # ═══════════════════════════════════════════════════════════════════════
    # DONE
    # ═══════════════════════════════════════════════════════════════════════
    total_time = time.time() - total_start
    
    logging.info("\n" + "=" * 60)
    logging.info("DAY 5 COMPLETE")
    logging.info("=" * 60)
    logging.info(f"  Total runtime: {total_time:.2f}s")
    logging.info(f"  Best model: {best_model_name}")
    logging.info(f"  Best F1 (weighted): {results_df.iloc[0]['F1 (Weighted)']:.4f}")
    logging.info(f"  Models saved to: {config.MODEL_DIR}/")
    logging.info(f"  Plots saved to:  {config.PLOT_DIR}/")
    logging.info(f"  Report saved to: {config.OUTPUT_DIR}/")
    logging.info(f"  Log saved to:    {config.LOG_DIR}/day05_experiment.log")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
