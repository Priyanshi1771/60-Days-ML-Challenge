"""
Day 2 - Breast Cancer Diagnosis
Model Training: Decision Tree with Pre-pruning, Post-pruning (CCP), and Visualization.

WHAT THIS MODULE TEACHES:
--------------------------
1. How Decision Trees actually work (splitting mechanics)
2. Pre-pruning: Constraining the tree DURING training (max_depth, min_samples)
3. Post-pruning (CCP): Growing a full tree, then pruning it back with cross-validation
4. Tree visualization: The killer feature of Decision Trees — full transparency
5. Gini vs Entropy: Two ways trees measure "impurity"

DECISION TREE DEEP DIVE:
--------------------------
At each node, the algorithm considers every feature and every possible threshold.
For each candidate split, it measures the "impurity" of the resulting child nodes.

GINI IMPURITY: Gini(node) = 1 - Σ(pᵢ²)
  - Probability of misclassifying a random sample
  - Range: 0 (pure) to 0.5 (maximum impurity for binary)
  - Example: node with 50/50 split → Gini = 1 - (0.5² + 0.5²) = 0.5

ENTROPY: H(node) = -Σ(pᵢ × log₂(pᵢ))
  - Information content (disorder/uncertainty)
  - Range: 0 (pure) to 1.0 (maximum entropy for binary)
  - Example: node with 50/50 split → H = -(0.5×log₂(0.5) + 0.5×log₂(0.5)) = 1.0

Both pick the split that maximizes INFORMATION GAIN = parent_impurity - weighted_child_impurity.
In practice, Gini and Entropy produce nearly identical trees.

PRE-PRUNING vs POST-PRUNING:
-------------------------------
PRE-PRUNING (constrain during growth):
  + Fast — tree never grows too large
  + Simple to implement (just set hyperparameters)
  - You might prune too aggressively and miss good splits deeper down
  - Hard to know the right constraints in advance

POST-PRUNING (grow full tree, then cut back):
  + More principled — uses the full tree's structure to decide what to cut
  + Cost-Complexity Pruning (CCP) has a single tunable parameter (alpha)
  + Can discover deep patterns that pre-pruning would block
  - Slower (must grow full tree first)

RECOMMENDATION: Start with CCP (post-pruning), use pre-pruning for speed in production.

HOW TO DEBUG DECISION TREES:
------------------------------
  - 100% train accuracy + low test accuracy = OVERFITTING → increase pruning
  - Low train AND test accuracy = UNDERFITTING → decrease pruning or add features
  - Tree depth > 15? Almost certainly overfitting unless you have millions of samples
  - Single feature dominates all splits? Check for data leakage or target encoding
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import logging
import os

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV, cross_validate
)
from sklearn.metrics import f1_score

from config import (
    RANDOM_SEED, VALIDATION_FOLDS, DT_PARAM_GRID,
    CCP_ALPHA_STEPS, MODEL_DIR, PLOT_DIR
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. UNPRUNED TREE (Baseline — shows overfitting)
# ============================================================
def train_unpruned_tree(X_train, y_train, X_test, y_test):
    """
    Train a fully grown (unpruned) Decision Tree.

    PURPOSE:
    This is intentionally a BAD model. We train it to DEMONSTRATE overfitting.
    An unpruned tree memorizes the training data by creating a leaf for nearly
    every training sample. This gives ~100% train accuracy but poor generalization.

    You should ALWAYS start with this baseline to understand:
    1. How deep the tree grows without constraints
    2. How badly it overfits
    3. How much pruning helps (by comparing pruned vs unpruned)

    ANALOGY: It's like studying for an exam by memorizing every practice question
    word-for-word. You'll ace the practice exam but fail any new questions.
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Unpruned Decision Tree (Overfitting Baseline)")
    logger.info("=" * 60)

    unpruned_tree = DecisionTreeClassifier(
        random_state=RANDOM_SEED
        # No max_depth, min_samples_split, etc. — let it grow fully
    )
    unpruned_tree.fit(X_train, y_train)

    train_acc = unpruned_tree.score(X_train, y_train)
    test_acc = unpruned_tree.score(X_test, y_test)
    gap = train_acc - test_acc

    logger.info(f"  Tree depth:     {unpruned_tree.get_depth()}")
    logger.info(f"  Number of leaves: {unpruned_tree.get_n_leaves()}")
    logger.info(f"  Train accuracy: {train_acc:.4f}")
    logger.info(f"  Test accuracy:  {test_acc:.4f}")
    logger.info(f"  Gap:            {gap:.4f} {'⚠️ OVERFITTING!' if gap > 0.05 else '✓ OK'}")

    return unpruned_tree


# ============================================================
# 2. PRE-PRUNED TREE (GridSearchCV)
# ============================================================
def train_prepruned_tree(X_train, y_train):
    """
    Train a Decision Tree with pre-pruning via GridSearchCV.

    PRE-PRUNING HYPERPARAMETERS EXPLAINED:
    ----------------------------------------
    max_depth (2, 3, 4, 5, ...):
      How many levels of questions the tree can ask.
      depth=2 means: question → question → prediction (at most 2 decisions)
      Lower depth = simpler model, less overfitting, but might miss patterns.

    min_samples_split (2, 5, 10, 20):
      A node must have at least this many samples to consider splitting.
      Higher = more conservative (avoids splitting small, noisy groups).

    min_samples_leaf (1, 2, 5, 10):
      Each leaf must contain at least this many samples.
      Higher = smoother predictions (no leaf represents a single outlier).

    criterion ('gini' vs 'entropy'):
      How the tree measures "impurity" at each node.
      Usually makes little practical difference.

    GRID SEARCH STRATEGY:
    We search all combinations and use stratified 10-fold CV.
    With 4 depths × 4 min_splits × 4 min_leafs × 2 criteria = 128 combinations,
    each evaluated with 10-fold CV = 1,280 tree fits. Still fast because
    Decision Trees train in milliseconds on 455 samples.
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Pre-pruned Decision Tree (GridSearchCV)")
    logger.info("=" * 60)

    base_dt = DecisionTreeClassifier(random_state=RANDOM_SEED)

    cv_strategy = StratifiedKFold(
        n_splits=VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    grid_search = GridSearchCV(
        estimator=base_dt,
        param_grid=DT_PARAM_GRID,
        cv=cv_strategy,
        scoring='f1',
        refit=True,
        return_train_score=True,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    logger.info(f"  Best parameters:")
    for param, value in best_params.items():
        logger.info(f"    {param:>20}: {value}")
    logger.info(f"  Best CV F1 Score: {grid_search.best_score_:.4f}")
    logger.info(f"  Tree depth:       {best_model.get_depth()}")
    logger.info(f"  Number of leaves: {best_model.get_n_leaves()}")

    # Show top 5 configurations
    results = grid_search.cv_results_
    top_indices = np.argsort(results['mean_test_score'])[-5:][::-1]
    logger.info(f"\n  Top 5 configurations:")
    for rank, idx in enumerate(top_indices, 1):
        params = results['params'][idx]
        train_score = results['mean_train_score'][idx]
        test_score = results['mean_test_score'][idx]
        std = results['std_test_score'][idx]
        logger.info(
            f"    #{rank}: F1={test_score:.4f}±{std:.4f} | "
            f"depth={params.get('max_depth', 'None'):>4}, "
            f"criterion={params['criterion']:>7}, "
            f"min_split={params['min_samples_split']:>2}, "
            f"min_leaf={params['min_samples_leaf']:>2} | "
            f"train={train_score:.4f}"
        )

    return best_model, grid_search


# ============================================================
# 3. COST-COMPLEXITY PRUNING (CCP) — The Principled Approach
# ============================================================
def train_ccp_pruned_tree(X_train, y_train):
    """
    Train a Decision Tree with Cost-Complexity Pruning (CCP).

    WHAT IS CCP?
    -------------
    CCP adds a penalty for tree complexity to the loss function:
        Total Cost = Classification Error + α × |number of leaves|

    Where α (ccp_alpha) controls the tradeoff:
    - α = 0: No penalty → fully grown tree (overfits)
    - α = large: Heavy penalty → tree is just a root node (underfits)
    - Optimal α: Found via cross-validation

    HOW IT WORKS:
    1. Grow a full tree (no constraints)
    2. Compute the "effective alpha" at which each subtree would be pruned
       (this is the cost_complexity_pruning_path)
    3. For each alpha value, prune the tree and measure CV performance
    4. Pick the alpha with the best CV score

    WHY CCP IS BETTER THAN PRE-PRUNING:
    CCP considers the WHOLE tree structure when deciding what to prune.
    Pre-pruning makes greedy local decisions (stopping growth at each node
    independently). CCP might keep a deep branch that pre-pruning would block,
    because that branch becomes valuable in the context of the full tree.

    ANALOGY: Pre-pruning is like deciding at each fork in the road whether
    to continue. CCP is like exploring the entire maze first, then removing
    dead-end paths.
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Cost-Complexity Pruned Tree (CCP)")
    logger.info("=" * 60)

    # Step 1: Get the pruning path (all possible alpha values)
    full_tree = DecisionTreeClassifier(random_state=RANDOM_SEED)
    full_tree.fit(X_train, y_train)

    path = full_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities

    logger.info(f"  Pruning path: {len(ccp_alphas)} alpha values")
    logger.info(f"  Alpha range: [{ccp_alphas.min():.6f}, {ccp_alphas.max():.6f}]")

    # Step 2: Cross-validate each alpha value
    cv_strategy = StratifiedKFold(
        n_splits=VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    # Sample alpha values if there are too many (for efficiency)
    if len(ccp_alphas) > CCP_ALPHA_STEPS:
        indices = np.linspace(0, len(ccp_alphas) - 1, CCP_ALPHA_STEPS, dtype=int)
        alpha_samples = ccp_alphas[indices]
    else:
        alpha_samples = ccp_alphas

    cv_scores_mean = []
    cv_scores_std = []
    train_scores_mean = []
    depths = []
    n_leaves = []

    for alpha in alpha_samples:
        dt = DecisionTreeClassifier(ccp_alpha=alpha, random_state=RANDOM_SEED)
        cv_results = cross_validate(
            dt, X_train, y_train,
            cv=cv_strategy, scoring='f1',
            return_train_score=True, n_jobs=-1
        )
        cv_scores_mean.append(cv_results['test_score'].mean())
        cv_scores_std.append(cv_results['test_score'].std())
        train_scores_mean.append(cv_results['train_score'].mean())

        dt.fit(X_train, y_train)
        depths.append(dt.get_depth())
        n_leaves.append(dt.get_n_leaves())

    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    train_scores_mean = np.array(train_scores_mean)

    # Step 3: Find optimal alpha
    best_idx = np.argmax(cv_scores_mean)
    best_alpha = alpha_samples[best_idx]

    logger.info(f"\n  Best alpha: {best_alpha:.6f}")
    logger.info(f"  Best CV F1: {cv_scores_mean[best_idx]:.4f} ± {cv_scores_std[best_idx]:.4f}")
    logger.info(f"  Tree depth at best alpha: {depths[best_idx]}")
    logger.info(f"  Leaves at best alpha: {n_leaves[best_idx]}")

    # Step 4: Train final model with best alpha
    best_ccp_tree = DecisionTreeClassifier(
        ccp_alpha=best_alpha,
        random_state=RANDOM_SEED
    )
    best_ccp_tree.fit(X_train, y_train)

    # Plot CCP path
    _plot_ccp_path(alpha_samples, train_scores_mean, cv_scores_mean,
                   cv_scores_std, best_alpha, depths)

    return best_ccp_tree, best_alpha, {
        'alphas': alpha_samples,
        'cv_scores': cv_scores_mean,
        'train_scores': train_scores_mean,
        'depths': depths,
        'n_leaves': n_leaves
    }


def _plot_ccp_path(alphas, train_scores, cv_scores, cv_std, best_alpha, depths):
    """
    Visualize the CCP pruning path.

    HOW TO READ THIS PLOT:
    -----------------------
    X-axis: alpha (pruning strength). Left = no pruning, Right = heavy pruning.
    Y-axis (left): F1 score. Blue = train, Orange = CV.

    The OPTIMAL alpha is where:
    - CV score is maximized (orange peak)
    - Train-CV gap is reasonable (no massive overfitting)

    LEFT SIDE (small alpha): Train score ≈ 1.0 but CV score is lower → overfitting
    RIGHT SIDE (large alpha): Both scores drop → underfitting (tree too simple)
    SWEET SPOT: Where CV score peaks
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Scores
    ax1.plot(alphas, train_scores, 'b-o', markersize=3, label='Train F1', alpha=0.7)
    ax1.plot(alphas, cv_scores, 'r-o', markersize=3, label='CV F1', alpha=0.7)
    ax1.fill_between(alphas, cv_scores - cv_std, cv_scores + cv_std,
                     alpha=0.15, color='red')
    ax1.axvline(x=best_alpha, color='green', linestyle='--', linewidth=2,
                label=f'Best α={best_alpha:.5f}')

    ax1.set_xlabel('Cost-Complexity Pruning Alpha (α)', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Tree depth on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(alphas, depths, 'g--', alpha=0.4, label='Tree Depth')
    ax2.set_ylabel('Tree Depth', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    plt.title('Cost-Complexity Pruning Path\n(Finding the optimal tree complexity)',
              fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(PLOT_DIR, "ccp_pruning_path.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  CCP pruning path saved to: {filepath}")


# ============================================================
# 4. TREE VISUALIZATION
# ============================================================
def visualize_tree(tree_model, feature_names, model_name="Decision Tree"):
    """
    Visualize the Decision Tree structure.

    THIS IS THE KILLER FEATURE OF DECISION TREES:
    ------------------------------------------------
    Unlike neural networks, SVMs, or ensemble methods, you can see
    EXACTLY how a Decision Tree makes every single prediction.

    Each node shows:
    - The splitting rule (e.g., "worst radius ≤ 16.795")
    - The Gini impurity or entropy
    - The number of samples reaching this node
    - The class distribution at this node
    - The predicted class (majority class)

    For medical applications, this transparency is invaluable.
    A doctor can trace the exact reasoning path and verify it
    makes clinical sense.

    COMMON MISTAKE: Trying to visualize an unpruned tree with 50+ nodes.
    It becomes an unreadable mess. Only visualize pruned/shallow trees.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    depth = tree_model.get_depth()
    n_leaves = tree_model.get_n_leaves()

    # Graphical visualization (for trees with reasonable depth)
    if depth <= 6:
        fig_width = max(20, n_leaves * 2.5)
        fig_height = max(10, depth * 2.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        plot_tree(
            tree_model,
            feature_names=feature_names,
            class_names=['Malignant', 'Benign'],
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax,
            proportion=True,
            impurity=True
        )

        ax.set_title(f'{model_name}\nDepth={depth}, Leaves={n_leaves}',
                     fontsize=14, fontweight='bold')

        filepath = os.path.join(PLOT_DIR, f"tree_visualization_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(filepath, dpi=120, bbox_inches='tight')
        plt.close()
        logger.info(f"  Tree visualization saved to: {filepath}")
    else:
        logger.info(f"  Tree too deep ({depth}) for graphical visualization. Using text instead.")

    # Text representation (always useful)
    text_tree = export_text(
        tree_model,
        feature_names=list(feature_names),
        max_depth=5  # Limit text output depth
    )
    logger.info(f"\n  Tree structure (text, max depth 5):\n{text_tree}")

    return text_tree


# ============================================================
# 5. DEPTH VS PERFORMANCE ANALYSIS
# ============================================================
def analyze_depth_performance(X_train, y_train):
    """
    Show how tree depth affects train vs test performance.

    THIS PLOT IS ONE OF THE MOST IMPORTANT IN ALL OF ML:
    It visually demonstrates the bias-variance tradeoff.

    - Low depth (1-2): HIGH BIAS — tree is too simple, both train and test are mediocre
    - Medium depth (3-5): SWEET SPOT — good train, good test
    - High depth (10+): HIGH VARIANCE — perfect train, poor test → overfitting

    This pattern repeats for EVERY model family (not just trees):
    - Logistic Regression: controlled by C (regularization)
    - Neural Networks: controlled by width/depth/dropout
    - KNN: controlled by k (neighbors)

    Understanding this plot deeply is FUNDAMENTAL to being an ML engineer.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    cv_strategy = StratifiedKFold(
        n_splits=VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    depths = range(1, 21)
    train_scores = []
    cv_scores = []
    cv_stds = []

    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_SEED)
        results = cross_validate(
            dt, X_train, y_train,
            cv=cv_strategy, scoring='f1',
            return_train_score=True, n_jobs=-1
        )
        train_scores.append(results['train_score'].mean())
        cv_scores.append(results['test_score'].mean())
        cv_stds.append(results['test_score'].std())

    train_scores = np.array(train_scores)
    cv_scores = np.array(cv_scores)
    cv_stds = np.array(cv_stds)

    best_depth = depths[np.argmax(cv_scores)]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(depths, train_scores, 'b-o', markersize=5, label='Train F1', linewidth=2)
    ax.plot(depths, cv_scores, 'r-o', markersize=5, label='CV F1', linewidth=2)
    ax.fill_between(depths, cv_scores - cv_stds, cv_scores + cv_stds,
                    alpha=0.15, color='red')

    ax.axvline(x=best_depth, color='green', linestyle='--', linewidth=2,
               label=f'Best Depth = {best_depth}')

    # Annotate regions
    ax.axvspan(1, 2.5, alpha=0.05, color='blue', label='Underfitting Zone')
    ax.axvspan(8, 20, alpha=0.05, color='red', label='Overfitting Zone')

    ax.set_xlabel('Tree Depth', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff: Tree Depth vs Performance\n'
                 '(The most important plot in ML)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 21))

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, "depth_vs_performance.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"\nDepth vs Performance plot saved to: {filepath}")
    logger.info(f"Optimal depth by CV: {best_depth}")

    return best_depth


# ============================================================
# 6. MODEL PERSISTENCE
# ============================================================
def save_model(model, filename):
    """Save trained model to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to: {filepath}")
    return filepath
