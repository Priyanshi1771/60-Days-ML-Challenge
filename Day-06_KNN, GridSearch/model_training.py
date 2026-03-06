"""
=============================================================================
 Day 6: Kidney Disease Prediction — Model Training
=============================================================================
 PRIMARY: KNN with exhaustive GridSearchCV hyperparameter tuning
 BASELINES: Logistic Regression, SVM, Random Forest, Decision Tree
 
 KEY LEARNING: GridSearchCV deep-dive
   - How to define parameter grids
   - Scoring strategies (F1 weighted for imbalanced data)
   - Extracting cv_results_ for analysis
   - Visualizing hyperparameter landscapes
   - Overfitting detection via train-test gap
=============================================================================
"""

import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score, learning_curve
)

import config

logger = logging.getLogger(__name__)


def train_knn_gridsearch(X_train, y_train):
    """
    Train KNN with exhaustive GridSearchCV.
    
    KNN HYPERPARAMETERS EXPLAINED:
    
    n_neighbors (K):
        - Small K (1-3): High variance, captures noise, complex boundaries
        - Large K (15-21): High bias, smooth boundaries, may miss patterns
        - Odd K avoids ties in binary classification
        
    weights:
        - "uniform": All K neighbors vote equally
        - "distance": Closer neighbors have more influence (1/distance)
        - Distance weighting often helps with overlapping classes
        
    metric:
        - "euclidean": Straight-line distance (L2 norm)
        - "manhattan": City-block distance (L1 norm) — robust to outliers
        - "minkowski": Generalized (p=1→manhattan, p=2→euclidean)
        
    WHY GRIDSEARCH?
    - Tests EVERY combination of hyperparameters
    - Uses cross-validation to estimate generalization performance
    - Prevents overfitting to a single train/test split
    - Computationally expensive but thorough for small datasets
    """
    logger.info("=" * 60)
    logger.info("TRAINING: KNN with GridSearchCV (Exhaustive)")
    logger.info("=" * 60)
    
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    
    # Full parameter grid
    param_grid = config.KNN_PARAM_GRID
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    logger.info(f"  Parameter grid: {total_combos} combinations")
    logger.info(f"  With {config.N_SPLITS}-fold CV: {total_combos * config.N_SPLITS} fits")
    
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        refit=True,
        n_jobs=-1,
        return_train_score=True,  # Track overfitting
        verbose=0
    )
    
    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start
    
    logger.info(f"\n  GridSearch completed in {elapsed:.2f}s")
    logger.info(f"  Best parameters:")
    for param, value in grid.best_params_.items():
        logger.info(f"    {param}: {value}")
    logger.info(f"  Best CV F1 (weighted): {grid.best_score_:.4f}")
    
    # Extract top 5 configurations
    results_df = pd.DataFrame(grid.cv_results_)
    results_df = results_df.sort_values("rank_test_score")
    
    logger.info(f"\n  Top 5 Configurations:")
    logger.info(f"  {'Rank':>4s} | {'K':>3s} | {'Weights':>10s} | {'Metric':>12s} | {'p':>2s} | {'CV F1':>8s} | {'Train F1':>9s} | {'Gap':>6s}")
    logger.info(f"  {'-'*70}")
    
    for _, row in results_df.head(5).iterrows():
        gap = row['mean_train_score'] - row['mean_test_score']
        logger.info(
            f"  {int(row['rank_test_score']):>4d} | "
            f"{str(row['param_n_neighbors']):>3s} | "
            f"{str(row['param_weights']):>10s} | "
            f"{str(row['param_metric']):>12s} | "
            f"{str(row['param_p']):>2s} | "
            f"{row['mean_test_score']:>8.4f} | "
            f"{row['mean_train_score']:>9.4f} | "
            f"{gap:>+6.4f}"
        )
    
    best_knn = grid.best_estimator_
    
    # ─── Visualize GridSearch results ────────────────────────────────────
    _plot_gridsearch_results(results_df, grid.best_params_)
    _plot_k_vs_performance(results_df, grid.best_params_)
    
    return best_knn, grid, results_df


def _plot_gridsearch_results(results_df, best_params):
    """Visualize the hyperparameter landscape from GridSearch."""
    
    # Plot: K vs F1 for each weight strategy
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for weight in ["uniform", "distance"]:
        ax = axes[0] if weight == "uniform" else axes[1]
        
        for metric in ["euclidean", "manhattan"]:
            mask = (results_df["param_weights"] == weight) & \
                   (results_df["param_metric"] == metric) & \
                   (results_df["param_p"] == best_params.get("p", 2))
            
            subset = results_df[mask].sort_values("param_n_neighbors")
            if len(subset) > 0:
                k_vals = subset["param_n_neighbors"].astype(int)
                test_scores = subset["mean_test_score"]
                train_scores = subset["mean_train_score"]
                
                ax.plot(k_vals, test_scores, 'o-', label=f'{metric} (test)', linewidth=2, markersize=6)
                ax.plot(k_vals, train_scores, 's--', label=f'{metric} (train)', alpha=0.5, linewidth=1)
        
        ax.set_title(f"Weights: {weight}", fontsize=13, fontweight='bold')
        ax.set_xlabel("K (n_neighbors)", fontsize=11)
        ax.set_ylabel("F1 Score (Weighted)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    
    fig.suptitle("GridSearch: K vs Performance by Weight Strategy & Metric",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_gridsearch_k_landscape.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_gridsearch_k_landscape.png")


def _plot_k_vs_performance(results_df, best_params):
    """Plot the bias-variance tradeoff as K changes."""
    
    # Average across all other params for each K
    k_summary = results_df.groupby("param_n_neighbors").agg({
        "mean_test_score": ["mean", "std"],
        "mean_train_score": ["mean", "std"]
    }).reset_index()
    k_summary.columns = ["K", "test_mean", "test_std", "train_mean", "train_std"]
    k_summary["K"] = k_summary["K"].astype(int)
    k_summary = k_summary.sort_values("K")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(k_summary["K"],
                     k_summary["train_mean"] - k_summary["train_std"],
                     k_summary["train_mean"] + k_summary["train_std"],
                     alpha=0.15, color="#FF7043")
    ax.fill_between(k_summary["K"],
                     k_summary["test_mean"] - k_summary["test_std"],
                     k_summary["test_mean"] + k_summary["test_std"],
                     alpha=0.15, color="#4FC3F7")
    
    ax.plot(k_summary["K"], k_summary["train_mean"], 'o-',
            color="#FF7043", linewidth=2.5, markersize=8, label="Train F1")
    ax.plot(k_summary["K"], k_summary["test_mean"], 's-',
            color="#4FC3F7", linewidth=2.5, markersize=8, label="CV Test F1")
    
    # Mark best K
    best_k = best_params["n_neighbors"]
    ax.axvline(x=best_k, color="#66BB6A", linestyle="--", linewidth=2, alpha=0.7,
               label=f"Best K={best_k}")
    
    ax.set_xlabel("K (Number of Neighbors)", fontsize=12)
    ax.set_ylabel("F1 Score (Weighted)", fontsize=12)
    ax.set_title("Bias-Variance Tradeoff: K vs Performance\n(averaged across all metric/weight combos)",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Annotate zones
    ax.annotate("High Variance\n(Overfitting)", xy=(2, ax.get_ylim()[0] + 0.01),
                fontsize=9, color='gray', style='italic')
    ax.annotate("High Bias\n(Underfitting)", xy=(max(k_summary["K"]) - 3, ax.get_ylim()[0] + 0.01),
                fontsize=9, color='gray', style='italic')
    
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_bias_variance_tradeoff.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_bias_variance_tradeoff.png")


def train_baselines(X_train, y_train):
    """Train baseline models for comparison against tuned KNN."""
    logger.info("=" * 60)
    logger.info("TRAINING: Baseline Models for Comparison")
    logger.info("=" * 60)
    
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    
    baselines = {}
    baseline_scores = {}
    
    models = {
        "Logistic Regression": GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
            {"C": config.LR_C_RANGE}, cv=cv, scoring="f1_weighted", n_jobs=-1
        ),
        "SVM (RBF)": GridSearchCV(
            SVC(kernel="rbf", random_state=config.RANDOM_SEED, probability=True),
            {"C": config.SVM_C_RANGE}, cv=cv, scoring="f1_weighted", n_jobs=-1
        ),
        "Random Forest": GridSearchCV(
            RandomForestClassifier(random_state=config.RANDOM_SEED, class_weight="balanced"),
            {"n_estimators": config.RF_N_ESTIMATORS}, cv=cv, scoring="f1_weighted", n_jobs=-1
        ),
        "Decision Tree": GridSearchCV(
            DecisionTreeClassifier(random_state=config.RANDOM_SEED, class_weight="balanced"),
            {"max_depth": [3, 5, 7, 10, None]}, cv=cv, scoring="f1_weighted", n_jobs=-1
        ),
        "KNN (default K=5)": KNeighborsClassifier(n_neighbors=5)
    }
    
    for name, model in models.items():
        start = time.time()
        if hasattr(model, 'best_score_'):
            model.fit(X_train, y_train)
            score = model.best_score_
            baselines[name] = model.best_estimator_
        else:
            model.fit(X_train, y_train)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
            score = scores.mean()
            baselines[name] = model
        
        elapsed = time.time() - start
        baseline_scores[name] = cross_val_score(
            baselines[name], X_train, y_train, cv=cv, scoring="f1_weighted"
        )
        logger.info(f"  {name:25s} | CV F1={baseline_scores[name].mean():.4f} ± {baseline_scores[name].std():.4f} | {elapsed:.2f}s")
    
    return baselines, baseline_scores


def save_models(best_knn, baselines, scaler, label_encoders, grid):
    """Save all models and preprocessing artifacts."""
    logger.info("-" * 60)
    logger.info("SAVING MODELS")
    logger.info("-" * 60)
    
    artifacts = {
        "best_knn": best_knn,
        "baselines": baselines,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "gridsearch_results": grid.cv_results_,
        "best_params": grid.best_params_
    }
    
    save_path = f"{config.MODEL_DIR}/day06_all_models.joblib"
    joblib.dump(artifacts, save_path)
    logger.info(f"  Saved to: {save_path}")
    return save_path
