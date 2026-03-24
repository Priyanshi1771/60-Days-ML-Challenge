"""
=============================================================================
 Day 7: Stroke Risk Prediction — Model Training
=============================================================================
 PRIMARY: XGBoost with SHAP interpretability
 BASELINES: Logistic Regression, Random Forest, SVM
 
 KEY LEARNING: SHAP (SHapley Additive exPlanations)
   - Global explanations: which features matter OVERALL
   - Local explanations: why THIS patient was predicted as stroke
   - Force plots, summary plots, dependence plots
   
 CRITICAL: Extreme imbalance (95:5) handled via scale_pos_weight
=============================================================================
"""
import logging, time, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.inspection import permutation_importance
import config

logger = logging.getLogger(__name__)

# Try XGBoost, fallback to sklearn
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
    logger.info("Using XGBoost")
except ImportError:
    HAS_XGBOOST = False
    logger.info("XGBoost unavailable — using sklearn GradientBoosting")

# Try SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def train_xgboost(X_train, y_train, X_train_unscaled=None):
    """
    Train XGBoost (or GradientBoosting) with GridSearchCV.
    
    WHY XGBoost FOR STROKE?
    - Handles imbalanced data natively via scale_pos_weight
    - Gradient boosting excels on tabular data
    - Built-in regularization (max_depth, learning_rate, subsample)
    - Feature importance + SHAP compatibility
    
    scale_pos_weight = n_negative / n_positive
    For 95:5 imbalance → scale_pos_weight ≈ 19
    This tells XGBoost to pay 19x more attention to stroke cases.
    """
    logger.info("=" * 60)
    logger.info(f"TRAINING: {'XGBoost' if HAS_XGBOOST else 'GradientBoosting'} with GridSearchCV")
    logger.info("=" * 60)

    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    if HAS_XGBOOST:
        base = XGBClassifier(random_state=config.RANDOM_SEED, eval_metric='logloss',
                             use_label_encoder=False, n_jobs=-1)
        param_grid = config.XGB_PARAM_GRID_FAST
    else:
        base = GradientBoostingClassifier(random_state=config.RANDOM_SEED)
        # Imbalance ratio for sample_weight approach
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        ratio = n_neg / n_pos if n_pos > 0 else 1
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
        }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    logger.info(f"  Grid: {total_combos} combos × {config.N_SPLITS}-fold = {total_combos * config.N_SPLITS} fits")

    # For GradientBoosting, use sample_weight for imbalance
    if not HAS_XGBOOST:
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        sample_weights = np.where(y_train == 1, n_neg / n_pos, 1.0)

    grid = GridSearchCV(base, param_grid, cv=cv, scoring="f1", refit=True, n_jobs=-1, return_train_score=True)

    start = time.time()
    if HAS_XGBOOST:
        grid.fit(X_train, y_train)
    else:
        grid.fit(X_train, y_train, sample_weight=sample_weights)
    elapsed = time.time() - start

    logger.info(f"  Completed in {elapsed:.1f}s")
    logger.info(f"  Best params: {grid.best_params_}")
    logger.info(f"  Best CV F1: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    return best_model, grid


def compute_shap_explanations(model, X_train, X_test, feature_names, y_test=None):
    """
    SHAP: SHapley Additive exPlanations
    
    WHAT IS SHAP?
    - Based on game theory (Shapley values from 1953)
    - Assigns each feature a contribution to the prediction
    - Positive SHAP = pushes toward stroke
    - Negative SHAP = pushes toward no stroke
    - Sum of all SHAP values = model output
    
    THREE TYPES OF SHAP PLOTS:
    1. Summary plot (beeswarm): Global feature importance + direction
    2. Bar plot: Mean absolute SHAP per feature
    3. Force plot: Single prediction explanation
    """
    logger.info("=" * 60)
    logger.info("SHAP INTERPRETABILITY ANALYSIS")
    logger.info("=" * 60)

    if HAS_SHAP:
        logger.info("  Computing SHAP values (TreeExplainer)...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # Summary plot (beeswarm)
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.title("🧠 SHAP Summary — Feature Impact on Stroke Prediction", fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{config.PLOT_DIR}/03_shap_summary.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("  Saved: 03_shap_summary.png")

            # Bar plot
            fig, ax = plt.subplots(figsize=(10, 7))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
            plt.title("⚡ Mean |SHAP| — Global Feature Importance", fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{config.PLOT_DIR}/04_shap_bar.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("  Saved: 04_shap_bar.png")

            # Top feature dependence plot
            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values
            top_feature_idx = np.abs(sv).mean(axis=0).argmax()
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(top_feature_idx, sv, X_test,
                                 feature_names=feature_names, show=False, ax=ax)
            plt.title(f"🔬 SHAP Dependence — {feature_names[top_feature_idx]}", fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{config.PLOT_DIR}/05_shap_dependence.png", dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("  Saved: 05_shap_dependence.png")

            return shap_values
        except Exception as e:
            logger.warning(f"  SHAP TreeExplainer failed: {e}")
            logger.info("  Falling back to permutation importance...")
            return _permutation_importance_fallback(model, X_test, y_test, feature_names)
    else:
        logger.info("  SHAP not installed — using permutation importance")
        return _permutation_importance_fallback(model, X_test, y_test, feature_names)


def _permutation_importance_fallback(model, X_test, y_test, feature_names):
    """Permutation importance as SHAP alternative."""
    logger.info("  Computing permutation importance...")

    result = permutation_importance(model, X_test, y_test, n_repeats=30,
                                    random_state=config.RANDOM_SEED, scoring='f1', n_jobs=-1)

    importances = result.importances_mean
    sorted_idx = np.argsort(importances)[::-1]

    logger.info(f"\n  Permutation Feature Importance (top 10):")
    for i in sorted_idx[:10]:
        logger.info(f"    {feature_names[i]:25s}: {importances[i]:.4f} ± {result.importances_std[i]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    top_n = min(15, len(feature_names))
    top_idx = sorted_idx[:top_n]
    colors = ['#EF5350' if importances[i] > 0.005 else '#4FC3F7' for i in top_idx]

    ax.barh(range(top_n), importances[top_idx], xerr=result.importances_std[top_idx],
            color=colors, alpha=0.85, capsize=4, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Permutation Importance (F1 decrease)", fontsize=11)
    ax.set_title("🧠 Feature Importance — Permutation Method\n(SHAP alternative: how much F1 drops when feature is shuffled)",
                 fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved: 03_feature_importance.png")

    # Built-in feature importance (tree-based)
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        fi_sorted = np.argsort(fi)[::-1]

        fig, ax = plt.subplots(figsize=(10, 7))
        top_idx2 = fi_sorted[:top_n]
        ax.barh(range(top_n), fi[top_idx2], color='#FF7043', alpha=0.85, edgecolor='white')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in top_idx2], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Gini / Split Importance", fontsize=11)
        ax.set_title("⚡ XGBoost Built-in Feature Importance (Gain)",
                     fontsize=13, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{config.PLOT_DIR}/04_builtin_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: 04_builtin_importance.png")

    return importances


def train_baselines(X_train, y_train):
    logger.info("=" * 60)
    logger.info("TRAINING: Baselines")
    logger.info("=" * 60)
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    weight_ratio = n_neg / n_pos

    baselines = {}
    scores = {}
    models_list = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED,
                                                   class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_SEED,
                                                 class_weight='balanced', n_jobs=-1),
        "SVM (RBF)": SVC(kernel='rbf', random_state=config.RANDOM_SEED, class_weight='balanced', probability=True),
    }
    for name, model in models_list.items():
        start = time.time()
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        baselines[name] = model
        scores[name] = cv_scores
        logger.info(f"  {name:25s} | CV F1={cv_scores.mean():.4f} ± {cv_scores.std():.4f} | {time.time()-start:.1f}s")

    return baselines, scores


def save_models(best_model, baselines, scaler, label_encoders, grid):
    save_path = f"{config.MODEL_DIR}/day07_all_models.joblib"
    joblib.dump({
        "best_xgb": best_model, "baselines": baselines,
        "scaler": scaler, "label_encoders": label_encoders,
        "best_params": grid.best_params_
    }, save_path)
    logger.info(f"  Models saved: {save_path}")
