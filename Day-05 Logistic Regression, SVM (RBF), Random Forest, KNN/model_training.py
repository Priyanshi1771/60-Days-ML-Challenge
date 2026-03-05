"""
=============================================================================
 Day 5: Thyroid Disease Classification — Model Training
=============================================================================
 Models:
   1. Gaussian Naive Bayes (baseline)
   2. Individual classifiers: LR, SVM, RF, KNN
   3. Hard Voting Ensemble
   4. Soft Voting Ensemble (probability-based — generally superior)
   5. Weighted Soft Voting Ensemble
 
 KEY LEARNING: Ensemble voting classifiers combine multiple diverse models
 to produce predictions that are often more robust than any single model.
 
 Hard Voting  = majority vote on predicted class labels
 Soft Voting  = average predicted probabilities, pick highest
 Weighted     = soft voting with per-model confidence weights
=============================================================================
"""

import logging
import time
import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

import config

logger = logging.getLogger(__name__)


def train_gaussian_nb(X_train, y_train):
    """
    Train Gaussian Naive Bayes with hyperparameter tuning.
    
    GNB assumes features follow a Gaussian distribution within each class.
    var_smoothing adds a portion of the largest variance to all variances
    for numerical stability — acts as a regularization parameter.
    
    WHY NB FOR THYROID?
    - Fast baseline that works well with few features
    - Handles the 3-class problem naturally
    - Strong independence assumption is reasonable for lab measurements
      (T3, T4, TSH are measured independently)
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Gaussian Naive Bayes (Baseline)")
    logger.info("=" * 60)
    
    # Grid search for var_smoothing
    param_grid = {"var_smoothing": config.GNB_VAR_SMOOTHING_RANGE}
    
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    
    grid = GridSearchCV(
        GaussianNB(),
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        refit=True,
        n_jobs=-1
    )
    
    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start
    
    best_gnb = grid.best_estimator_
    logger.info(f"  Best var_smoothing: {grid.best_params_['var_smoothing']}")
    logger.info(f"  Best CV F1 (weighted): {grid.best_score_:.4f}")
    logger.info(f"  Training time: {elapsed:.3f}s")
    
    # Cross-validation scores for reporting
    cv_scores = cross_val_score(best_gnb, X_train, y_train, cv=cv, scoring="f1_weighted")
    logger.info(f"  CV F1 scores: {cv_scores.round(4)}")
    logger.info(f"  CV F1 mean ± std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return best_gnb, cv_scores


def train_individual_classifiers(X_train, y_train):
    """
    Train individual classifiers that will form the voting ensemble.
    
    DIVERSITY IS KEY: Ensemble methods work best when base classifiers
    make different kinds of errors. We deliberately choose models with
    different learning paradigms:
    
    - Logistic Regression: Linear decision boundaries
    - SVM (RBF kernel): Non-linear, margin-based
    - Random Forest: Tree-based, handles interactions
    - KNN: Instance-based, local decision boundaries
    
    Each is tuned independently via GridSearchCV before ensembling.
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Individual Classifiers for Ensemble")
    logger.info("=" * 60)
    
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    
    classifiers = {}
    cv_scores_dict = {}
    
    # ─── 1. Logistic Regression ──────────────────────────────────────────
    logger.info("\n[1/4] Logistic Regression")
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=config.LR_MAX_ITER, random_state=config.RANDOM_SEED),
        {"C": config.LR_C_RANGE},
        cv=cv, scoring="f1_weighted", refit=True, n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    classifiers["lr"] = lr_grid.best_estimator_
    cv_scores_dict["lr"] = cross_val_score(lr_grid.best_estimator_, X_train, y_train, cv=cv, scoring="f1_weighted")
    logger.info(f"  Best C={lr_grid.best_params_['C']} | CV F1={cv_scores_dict['lr'].mean():.4f} ± {cv_scores_dict['lr'].std():.4f}")
    
    # ─── 2. SVM ──────────────────────────────────────────────────────────
    logger.info("\n[2/4] SVM (RBF kernel)")
    # SVM needs probability=True for soft voting
    svm_grid = GridSearchCV(
        SVC(kernel=config.SVM_KERNEL, random_state=config.RANDOM_SEED, probability=True),
        {"C": config.SVM_C_RANGE},
        cv=cv, scoring="f1_weighted", refit=True, n_jobs=-1
    )
    svm_grid.fit(X_train, y_train)
    classifiers["svm"] = svm_grid.best_estimator_
    cv_scores_dict["svm"] = cross_val_score(svm_grid.best_estimator_, X_train, y_train, cv=cv, scoring="f1_weighted")
    logger.info(f"  Best C={svm_grid.best_params_['C']} | CV F1={cv_scores_dict['svm'].mean():.4f} ± {cv_scores_dict['svm'].std():.4f}")
    
    # ─── 3. Random Forest ────────────────────────────────────────────────
    logger.info("\n[3/4] Random Forest")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=config.RANDOM_SEED, class_weight='balanced'),
        {"n_estimators": config.RF_N_ESTIMATORS, "max_depth": config.RF_MAX_DEPTH},
        cv=cv, scoring="f1_weighted", refit=True, n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    classifiers["rf"] = rf_grid.best_estimator_
    cv_scores_dict["rf"] = cross_val_score(rf_grid.best_estimator_, X_train, y_train, cv=cv, scoring="f1_weighted")
    logger.info(f"  Best params={rf_grid.best_params_} | CV F1={cv_scores_dict['rf'].mean():.4f} ± {cv_scores_dict['rf'].std():.4f}")
    
    # ─── 4. KNN ──────────────────────────────────────────────────────────
    logger.info("\n[4/4] K-Nearest Neighbors")
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        {"n_neighbors": config.KNN_N_NEIGHBORS},
        cv=cv, scoring="f1_weighted", refit=True, n_jobs=-1
    )
    knn_grid.fit(X_train, y_train)
    classifiers["knn"] = knn_grid.best_estimator_
    cv_scores_dict["knn"] = cross_val_score(knn_grid.best_estimator_, X_train, y_train, cv=cv, scoring="f1_weighted")
    logger.info(f"  Best k={knn_grid.best_params_['n_neighbors']} | CV F1={cv_scores_dict['knn'].mean():.4f} ± {cv_scores_dict['knn'].std():.4f}")
    
    return classifiers, cv_scores_dict


def train_voting_ensembles(X_train, y_train, classifiers):
    """
    Build and train Voting Ensemble classifiers.
    
    THREE TYPES OF VOTING:
    
    1. HARD VOTING (majority vote):
       - Each classifier votes for a class
       - Final prediction = most common vote
       - Simple but effective; doesn't use prediction confidence
       
    2. SOFT VOTING (probability averaging):
       - Each classifier outputs probability per class
       - Averages probabilities across all classifiers
       - Final prediction = class with highest average probability
       - Usually BETTER than hard voting (uses more information)
       - REQUIRES: all classifiers support predict_proba()
       
    3. WEIGHTED SOFT VOTING:
       - Same as soft voting but weights classifiers by their CV performance
       - Gives more influence to better individual models
       - Weights derived from cross-validation F1 scores
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Voting Ensemble Classifiers")
    logger.info("=" * 60)
    
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    
    estimators = [
        ("lr", classifiers["lr"]),
        ("svm", classifiers["svm"]),
        ("rf", classifiers["rf"]),
        ("knn", classifiers["knn"])
    ]
    
    ensembles = {}
    ensemble_cv_scores = {}
    
    # ─── 1. Hard Voting ──────────────────────────────────────────────────
    logger.info("\n[1/3] Hard Voting Ensemble")
    hard_vote = VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)
    hard_vote.fit(X_train, y_train)
    hard_cv = cross_val_score(hard_vote, X_train, y_train, cv=cv, scoring="f1_weighted")
    ensembles["hard_voting"] = hard_vote
    ensemble_cv_scores["hard_voting"] = hard_cv
    logger.info(f"  CV F1: {hard_cv.mean():.4f} ± {hard_cv.std():.4f}")
    
    # ─── 2. Soft Voting ──────────────────────────────────────────────────
    logger.info("\n[2/3] Soft Voting Ensemble")
    soft_vote = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    soft_vote.fit(X_train, y_train)
    soft_cv = cross_val_score(soft_vote, X_train, y_train, cv=cv, scoring="f1_weighted")
    ensembles["soft_voting"] = soft_vote
    ensemble_cv_scores["soft_voting"] = soft_cv
    logger.info(f"  CV F1: {soft_cv.mean():.4f} ± {soft_cv.std():.4f}")
    
    # ─── 3. Weighted Soft Voting ─────────────────────────────────────────
    logger.info("\n[3/3] Weighted Soft Voting Ensemble")
    
    # Compute weights from individual CV scores
    individual_scores = []
    for name, clf in estimators:
        score = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_weighted").mean()
        individual_scores.append(score)
    
    # Normalize weights to sum to 1
    weights = np.array(individual_scores)
    weights = weights / weights.sum()
    
    logger.info(f"  Weights: LR={weights[0]:.3f}, SVM={weights[1]:.3f}, RF={weights[2]:.3f}, KNN={weights[3]:.3f}")
    
    weighted_vote = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=weights.tolist(),
        n_jobs=-1
    )
    weighted_vote.fit(X_train, y_train)
    weighted_cv = cross_val_score(weighted_vote, X_train, y_train, cv=cv, scoring="f1_weighted")
    ensembles["weighted_voting"] = weighted_vote
    ensemble_cv_scores["weighted_voting"] = weighted_cv
    logger.info(f"  CV F1: {weighted_cv.mean():.4f} ± {weighted_cv.std():.4f}")
    
    return ensembles, ensemble_cv_scores, weights


def save_models(gnb, classifiers, ensembles, scaler, label_encoder, weights):
    """Save all models, scaler, and encoder for reproducibility."""
    logger.info("-" * 60)
    logger.info("SAVING MODELS")
    logger.info("-" * 60)
    
    artifacts = {
        "gaussian_nb": gnb,
        "logistic_regression": classifiers["lr"],
        "svm": classifiers["svm"],
        "random_forest": classifiers["rf"],
        "knn": classifiers["knn"],
        "hard_voting_ensemble": ensembles["hard_voting"],
        "soft_voting_ensemble": ensembles["soft_voting"],
        "weighted_voting_ensemble": ensembles["weighted_voting"],
        "scaler": scaler,
        "label_encoder": label_encoder,
        "ensemble_weights": weights
    }
    
    save_path = f"{config.MODEL_DIR}/day05_all_models.joblib"
    joblib.dump(artifacts, save_path)
    logger.info(f"  All models saved to: {save_path}")
    
    return save_path
