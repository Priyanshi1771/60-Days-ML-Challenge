"""
=============================================================================
 Day 8: Anemia Detection — Model Training
=============================================================================
 PRIMARY: AdaBoost with scaling + outlier removal ablation study
 
 KEY LEARNING: AdaBoost (Adaptive Boosting)
   - Sequentially trains weak learners (decision stumps)
   - Each new learner focuses on MISTAKES of the previous one
   - Misclassified samples get HIGHER weights → harder to ignore
   - Final prediction = weighted vote of all weak learners
   
 ABLATION STUDY:
   - Test 4 outlier methods × 4 scaling methods = 16 combos
   - Shows which preprocessing pipeline is optimal for AdaBoost
=============================================================================
"""
import logging, time, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
import config
from data_pipeline import detect_outliers, get_scaler

logger = logging.getLogger(__name__)


def run_ablation_study(X_train_raw, y_train_raw):
    """
    ABLATION STUDY: Test all combinations of outlier removal + scaling.
    
    WHY THIS MATTERS:
    - AdaBoost uses Decision Stumps (1-level trees) as weak learners
    - Stumps split on single thresholds → outliers can shift split points
    - Scaling doesn't affect tree splits BUT affects the outlier removal step
    - RobustScaler uses median/IQR → naturally downweights outlier influence
    
    We test every combo to find the optimal preprocessing pipeline.
    """
    logger.info("=" * 60)
    logger.info("ABLATION STUDY: Outlier × Scaling Combinations")
    logger.info("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    results = []

    for outlier_method in config.OUTLIER_METHODS:
        # Apply outlier removal on raw training data
        X_clean, y_clean, n_removed = detect_outliers(X_train_raw.copy(), y_train_raw.copy(), outlier_method)
        
        for scale_method in config.SCALING_METHODS:
            scaler = get_scaler(scale_method)
            X_scaled = scaler.fit_transform(X_clean) if scaler else X_clean.copy()

            ada = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=100, learning_rate=0.1, random_state=config.RANDOM_SEED)
            
            scores = cross_val_score(ada, X_scaled, y_clean, cv=cv, scoring="f1_weighted")
            
            results.append({
                "Outlier Method": outlier_method,
                "Scaling Method": scale_method,
                "CV F1 Mean": scores.mean(),
                "CV F1 Std": scores.std(),
                "Samples Removed": n_removed,
                "Samples Used": len(y_clean)
            })
            logger.info(f"  {outlier_method:20s} + {scale_method:10s} → F1={scores.mean():.4f} ± {scores.std():.4f} "
                         f"(removed {n_removed}, used {len(y_clean)})")

    results_df = pd.DataFrame(results).sort_values("CV F1 Mean", ascending=False)
    
    # Find best combo
    best = results_df.iloc[0]
    logger.info(f"\n  🏆 Best combo: {best['Outlier Method']} + {best['Scaling Method']}")
    logger.info(f"     F1 = {best['CV F1 Mean']:.4f} ± {best['CV F1 Std']:.4f}")

    _plot_ablation_heatmap(results_df)
    return results_df, best["Outlier Method"], best["Scaling Method"]


def _plot_ablation_heatmap(results_df):
    """Visualize ablation study as a heatmap."""
    pivot = results_df.pivot_table(
        values="CV F1 Mean", index="Outlier Method", columns="Scaling Method")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns_map = plt.cm.YlOrRd
    im = ax.imshow(pivot.values, cmap=sns_map, aspect='auto', vmin=pivot.values.min()-0.01, vmax=pivot.values.max()+0.01)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val > pivot.values.mean() else 'black'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center', fontweight='bold', fontsize=10, color=color)

    # Mark best cell
    best_idx = np.unravel_index(pivot.values.argmax(), pivot.values.shape)
    rect = plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1, linewidth=3, edgecolor='#00E676', facecolor='none')
    ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label="CV F1 (Weighted)")
    ax.set_xlabel("Scaling Method", fontsize=12, fontweight='bold')
    ax.set_ylabel("Outlier Removal Method", fontsize=12, fontweight='bold')
    ax.set_title("🧪 Ablation Study: Outlier × Scaling Impact on AdaBoost\n(green border = best combination)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_ablation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_ablation_heatmap.png")


import seaborn as sns


def train_adaboost(X_train, y_train, outlier_method, scale_method):
    """
    Train AdaBoost with best preprocessing from ablation study.
    
    ADABOOST EXPLAINED:
    
    Round 1: Train stump on original data
    Round 2: Increase weight of misclassified samples → train new stump
    Round 3: Even more weight on still-wrong samples → train new stump
    ...
    Round N: Weighted vote of all stumps = final prediction
    
    Key hyperparameters:
    - n_estimators: Number of weak learners (more = more complex)
    - learning_rate: Shrinks contribution of each learner (lower = more robust)
    - Interaction: n_estimators × learning_rate (more estimators needs lower LR)
    """
    logger.info("=" * 60)
    logger.info("TRAINING: AdaBoost with GridSearchCV")
    logger.info("=" * 60)

    # Apply best preprocessing
    X_clean, y_clean, n_removed = detect_outliers(X_train.copy(), y_train.copy(), outlier_method)
    scaler = get_scaler(scale_method)
    X_scaled = scaler.fit_transform(X_clean) if scaler else X_clean.copy()
    logger.info(f"  Preprocessing: {outlier_method} + {scale_method} (removed {n_removed} outliers)")

    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    grid = GridSearchCV(
        AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                           random_state=config.RANDOM_SEED),
        config.ADA_PARAM_GRID, cv=cv, scoring="f1_weighted",
        refit=True, n_jobs=-1, return_train_score=True)

    start = time.time()
    grid.fit(X_scaled, y_clean)
    elapsed = time.time() - start

    logger.info(f"  Completed in {elapsed:.1f}s")
    logger.info(f"  Best params: {grid.best_params_}")
    logger.info(f"  Best CV F1: {grid.best_score_:.4f}")

    best_ada = grid.best_estimator_

    # Plot learning rate vs n_estimators landscape
    _plot_adaboost_landscape(grid)

    return best_ada, grid, scaler, X_clean, y_clean, X_scaled


def _plot_adaboost_landscape(grid):
    """Visualize n_estimators × learning_rate performance."""
    results = pd.DataFrame(grid.cv_results_)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: n_estimators vs F1 for each learning_rate
    for lr in config.ADA_PARAM_GRID["learning_rate"]:
        mask = results["param_learning_rate"] == lr
        subset = results[mask].sort_values("param_n_estimators")
        axes[0].plot(subset["param_n_estimators"].astype(int), subset["mean_test_score"],
                     'o-', label=f'LR={lr}', linewidth=2, markersize=6)
    axes[0].set_xlabel("n_estimators", fontsize=11)
    axes[0].set_ylabel("CV F1 (Weighted)", fontsize=11)
    axes[0].set_title("⚡ AdaBoost: n_estimators vs Performance", fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].spines[['top', 'right']].set_visible(False)

    # Right: Train vs Test to detect overfitting
    best_lr = grid.best_params_["learning_rate"]
    mask = results["param_learning_rate"] == best_lr
    subset = results[mask].sort_values("param_n_estimators")
    axes[1].plot(subset["param_n_estimators"].astype(int), subset["mean_train_score"],
                 's--', color='#FF7043', linewidth=2, label='Train F1')
    axes[1].plot(subset["param_n_estimators"].astype(int), subset["mean_test_score"],
                 'o-', color='#4FC3F7', linewidth=2, label='CV Test F1')
    axes[1].fill_between(subset["param_n_estimators"].astype(int),
                          subset["mean_test_score"] - subset["std_test_score"],
                          subset["mean_test_score"] + subset["std_test_score"],
                          alpha=0.15, color='#4FC3F7')
    axes[1].set_xlabel("n_estimators", fontsize=11)
    axes[1].set_ylabel("F1 (Weighted)", fontsize=11)
    axes[1].set_title(f"🔍 Overfitting Check (LR={best_lr})", fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_adaboost_landscape.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_adaboost_landscape.png")


def train_baselines(X_train_scaled, y_train):
    logger.info("=" * 60)
    logger.info("TRAINING: Baselines")
    logger.info("=" * 60)
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    baselines, scores = {}, {}
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_SEED, n_jobs=-1),
        "SVM (RBF)": SVC(kernel='rbf', random_state=config.RANDOM_SEED, probability=True),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=config.RANDOM_SEED),
    }
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        cv_s = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1_weighted")
        baselines[name] = model
        scores[name] = cv_s
        logger.info(f"  {name:25s} | CV F1={cv_s.mean():.4f} ± {cv_s.std():.4f}")

    return baselines, scores


def save_models(best_ada, baselines, scaler, grid):
    save_path = f"{config.MODEL_DIR}/day08_all_models.joblib"
    joblib.dump({"best_adaboost": best_ada, "baselines": baselines, "scaler": scaler,
                 "best_params": grid.best_params_}, save_path)
    logger.info(f"  Models saved: {save_path}")
