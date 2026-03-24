"""
=============================================================================
 Day 9: Hepatitis Diagnosis — Model Training (Optimized)
=============================================================================
 PRIMARY: Perceptron — the simplest neural network (single layer, no hidden)
 FOCUS: ROC curve analysis + threshold optimization
 
 WHY PERCEPTRON FOR DAY 9?
 - Foundation for all neural networks (Day 10+ enters deep learning)
 - Shows limitations of linear classifiers on medical data
 - Perfect vehicle for ROC analysis: vary threshold → trace entire curve
 - Historically important: Rosenblatt (1958) → modern deep learning
 
 PERCEPTRON vs LOGISTIC REGRESSION:
 - Perceptron: hinge-like loss, no probabilities, online learning
 - LR: log loss, calibrated probabilities, batch optimization
 - Perceptron needs CalibratedClassifierCV to produce probabilities for ROC
=============================================================================
"""
import logging, time, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import config

logger = logging.getLogger(__name__)


def train_perceptron(X_train, y_train):
    """
    Train Perceptron with GridSearchCV + probability calibration.
    
    THE PERCEPTRON ALGORITHM:
    1. Initialize weights to small random values
    2. For each sample: predict = sign(w·x + b)
    3. If wrong: w += η × y × x  (nudge weights toward correct answer)
    4. If correct: do nothing
    5. Repeat for max_iter epochs
    
    LIMITATION: Only learns linearly separable patterns.
    If Die/Live aren't separable by a hyperplane → Perceptron fails.
    This motivates Day 10's move to deep learning.
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Perceptron + Calibration")
    logger.info("=" * 60)

    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    # GridSearch
    grid = GridSearchCV(
        Perceptron(random_state=config.RANDOM_SEED),
        config.PERCEPTRON_PARAM_GRID, cv=cv, scoring="roc_auc",  # Optimize for AUC!
        refit=True, n_jobs=-1, return_train_score=True)

    start = time.time()
    grid.fit(X_train, y_train)
    logger.info(f"  GridSearch: {time.time()-start:.1f}s | Best AUC: {grid.best_score_:.4f}")
    logger.info(f"  Best params: {grid.best_params_}")

    best_perceptron = grid.best_estimator_

    # Calibrate to get probabilities (Perceptron doesn't output them natively)
    # CalibratedClassifierCV wraps the model + learns probability mapping via Platt scaling
    calibrated = CalibratedClassifierCV(best_perceptron, cv=cv, method='sigmoid')
    calibrated.fit(X_train, y_train)
    logger.info("  Calibrated with Platt scaling (sigmoid) for probability output")

    # Visualize perceptron weights
    _plot_perceptron_weights(best_perceptron)

    return best_perceptron, calibrated, grid


def _plot_perceptron_weights(model):
    """Visualize what the Perceptron learned (weight vector)."""
    weights = model.coef_.ravel()
    sorted_idx = np.argsort(np.abs(weights))[::-1]
    names = [config.FEATURE_NAMES[i] for i in sorted_idx]
    vals = weights[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#EF5350' if v < 0 else '#66BB6A' for v in vals]
    ax.barh(range(len(names)), vals, color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(x=0, color='gray', linewidth=1, linestyle='--')
    ax.set_xlabel("Perceptron Weight", fontsize=11)
    ax.set_title("🧬 Perceptron Weights — What the Model Learned\n"
                 "(Green = pushes toward LIVE | Red = pushes toward DIE)",
                 fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_perceptron_weights.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_perceptron_weights.png")


def train_baselines(X_train, y_train):
    """Train baselines — all optimized with n_jobs=-1 where possible."""
    logger.info("=" * 60)
    logger.info("TRAINING: Baselines")
    logger.info("=" * 60)
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    baselines = {}
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=config.RANDOM_SEED,
                                                   class_weight='balanced', solver='lbfgs'),
        "SVM (RBF)": SVC(kernel='rbf', random_state=config.RANDOM_SEED,
                          class_weight='balanced', probability=True, cache_size=200),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=config.RANDOM_SEED,
                                                 class_weight='balanced', n_jobs=-1, max_depth=7),
        "MLP (1 hidden)": MLPClassifier(hidden_layer_sizes=(32,), max_iter=500,
                                         random_state=config.RANDOM_SEED, early_stopping=True),
    }
    for name, model in models.items():
        t = time.time()
        model.fit(X_train, y_train)
        cv_s = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        baselines[name] = model
        logger.info(f"  {name:25s} | CV AUC={cv_s.mean():.4f} ± {cv_s.std():.4f} | {time.time()-t:.1f}s")

    return baselines


def save_models(perceptron, calibrated, baselines, scaler, imputer, grid):
    path = f"{config.MODEL_DIR}/day09_models.joblib"
    joblib.dump({"perceptron": perceptron, "calibrated": calibrated,
                 "baselines": baselines, "scaler": scaler, "imputer": imputer,
                 "best_params": grid.best_params_}, path, compress=3)
    logger.info(f"  Saved (compressed): {path}")
