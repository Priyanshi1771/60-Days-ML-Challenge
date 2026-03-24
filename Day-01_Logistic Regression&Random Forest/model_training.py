"""
Day 1 - Heart Disease Prediction
Model Training: Logistic Regression (primary) + Random Forest (comparison).

WHAT THIS MODULE DOES:
----------------------
Trains and validates models using proper cross-validation, hyperparameter
tuning, and model persistence (saving/loading).

KEY CONCEPTS:
-------------
1. Cross-Validation: Why a single train/test split isn't enough
   - With only 303 samples, a single split might give you a "lucky" or "unlucky"
     test set. CV averages over multiple splits for a more reliable estimate.
   - We use STRATIFIED K-Fold to maintain class proportions in each fold.

2. Hyperparameter Tuning: Finding the best C value
   - C controls the tradeoff between fitting the training data well and keeping
     the model simple (regularization). Too large C = overfitting, too small = underfitting.
   - We search over a grid of C values and pick the one with the best CV score.

3. Model Persistence: Saving trained models
   - After finding the best model, save it so you don't retrain every time.
   - In production, you'd load this saved model to make predictions on new patients.

HOW TO DEBUG:
-------------
  - If CV scores have HIGH VARIANCE (e.g., 0.65, 0.90, 0.72, 0.85, 0.78),
    your model is unstable. Try more regularization (smaller C) or more data.
  - If train accuracy >> test accuracy, you're overfitting.
  - If both train and test accuracy are LOW, you're underfitting (try higher C or more features).
"""

import numpy as np
import joblib
import logging
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, f1_score

from config import (
    RANDOM_SEED, VALIDATION_FOLDS, LR_C_VALUES,
    LR_MAX_ITER, LR_SOLVER, RF_N_ESTIMATORS, RF_MAX_DEPTH,
    MODEL_DIR
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. LOGISTIC REGRESSION WITH CROSS-VALIDATED TUNING
# ============================================================
def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model with GridSearchCV for hyperparameter tuning.

    WHY LOGISTIC REGRESSION?
    -------------------------
    Despite its name, Logistic Regression is a CLASSIFICATION algorithm.
    It models the probability of the positive class using the sigmoid function:
        P(y=1|x) = 1 / (1 + exp(-(w·x + b)))

    The "logistic" comes from the logistic (sigmoid) function, not from "regression."

    HOW IT WORKS (Intuition):
    --------------------------
    1. Compute a weighted sum of features: z = w1*age + w2*chol + ... + b
    2. Pass through sigmoid to get a probability: P = sigmoid(z)
    3. If P > 0.5, predict "heart disease"; otherwise "healthy"
    4. The weights (w1, w2, ...) are learned by minimizing log loss.

    WHAT IS C?
    ----------
    C is the INVERSE of regularization strength.
    - C = 0.001: Very strong regularization → simple model, might underfit
    - C = 1.0:   Moderate regularization → good default
    - C = 100.0: Very weak regularization → complex model, might overfit

    Think of regularization as a "skepticism penalty." It says: "I don't fully
    trust the training data, so I'll keep my weights small (close to 0) unless
    the evidence is overwhelming."

    WHY L2 (Ridge) REGULARIZATION?
    --------------------------------
    L2 shrinks all coefficients toward zero but never makes them exactly zero.
    This is fine when all 13 features are potentially relevant (as in heart disease).
    If you suspected many features were irrelevant, you'd use L1 (Lasso), which
    can zero out features entirely (feature selection).
    """

    logger.info("=" * 60)
    logger.info("TRAINING: Logistic Regression with GridSearchCV")
    logger.info("=" * 60)

    # Define the parameter grid
    param_grid = {
        'C': LR_C_VALUES,
        'penalty': ['l2'],    # L2 regularization (Ridge)
        'solver': [LR_SOLVER]
    }

    # Create the base model
    base_lr = LogisticRegression(
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_SEED,
        # class_weight='balanced'  # Uncomment if classes are imbalanced
        # 'balanced' auto-adjusts weights inversely proportional to class frequencies
    )

    # Stratified K-Fold: each fold has approximately the same % of positive cases
    cv_strategy = StratifiedKFold(
        n_splits=VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    # GridSearchCV: tries every combination and picks the best via CV
    # scoring='f1' because accuracy can be misleading with even mild imbalance
    grid_search = GridSearchCV(
        estimator=base_lr,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='f1',          # Primary metric for selection
        refit=True,            # Refit best model on full training set
        return_train_score=True,  # To check for overfitting
        verbose=0,
        n_jobs=-1              # Use all CPU cores
    )

    grid_search.fit(X_train, y_train)

    # Extract results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info(f"Best C value: {best_params['C']}")
    logger.info(f"Best CV F1 Score: {best_score:.4f}")

    # Log all C values and their scores (useful for understanding the landscape)
    results = grid_search.cv_results_
    logger.info("\nC Value Landscape:")
    for i, c_val in enumerate(LR_C_VALUES):
        mean_train = results['mean_train_score'][i]
        mean_test = results['mean_test_score'][i]
        std_test = results['std_test_score'][i]
        gap = mean_train - mean_test

        logger.info(
            f"  C={c_val:>8.3f} | Train F1: {mean_train:.4f} | "
            f"Val F1: {mean_test:.4f} ± {std_test:.4f} | "
            f"Gap: {gap:.4f} {'⚠️ OVERFITTING' if gap > 0.1 else '✓'}"
        )

    # Feature importance from coefficients
    # In Logistic Regression, the coefficient magnitude tells you how much
    # each feature contributes to the prediction (after scaling).
    # Positive coefficient = increases probability of heart disease.
    # Negative coefficient = decreases probability.
    logger.info("\nModel Coefficients (Feature Importance):")
    coefficients = best_model.coef_[0]

    return best_model, grid_search


# ============================================================
# 2. RANDOM FOREST (Comparison Baseline)
# ============================================================
def train_random_forest(X_train, y_train):
    """
    Train a Random Forest as a comparison model.

    WHY COMPARE?
    -------------
    Random Forest captures nonlinear relationships that Logistic Regression can't.
    If RF dramatically outperforms LR, it suggests the data has nonlinear patterns
    that a linear model misses.

    If both perform similarly, prefer the simpler model (LR) because:
    1. It's more interpretable (doctors can understand it)
    2. It's faster to deploy
    3. Simpler models generalize better (Occam's razor)

    NOTE: RF doesn't need scaled features! Trees split on thresholds, not distances.
    That's why we pass X_train_raw (unscaled) for tree-based models.
    """

    logger.info("=" * 60)
    logger.info("TRAINING: Random Forest (Comparison Baseline)")
    logger.info("=" * 60)

    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # Quick cross-validation to get a performance estimate
    cv_strategy = StratifiedKFold(
        n_splits=VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    cv_scores = cross_val_score(
        rf_model, X_train, y_train,
        cv=cv_strategy, scoring='f1', n_jobs=-1
    )

    logger.info(f"RF CV F1 Scores: {cv_scores.round(4)}")
    logger.info(f"RF CV F1 Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit on full training set
    rf_model.fit(X_train, y_train)

    return rf_model, cv_scores


# ============================================================
# 3. MODEL PERSISTENCE
# ============================================================
def save_model(model, filename):
    """
    Save a trained model to disk using joblib.

    WHY SAVE MODELS?
    -----------------
    Training takes time (especially for deep learning later in your 60 days).
    Save the trained model so you can:
    1. Load it later without retraining
    2. Deploy it in a production system
    3. Version control your experiments

    COMMON MISTAKE: Only saving the model, not the preprocessing artifacts
    (scaler, encoder). When you load the model for prediction, you ALSO need
    the exact same scaler that was fit on training data!
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to: {filepath}")
    return filepath


def load_model(filename):
    """Load a trained model from disk."""
    filepath = os.path.join(MODEL_DIR, filename)
    model = joblib.load(filepath)
    logger.info(f"Model loaded from: {filepath}")
    return model
