"""
=============================================================================
 Day 6: Kidney Disease Prediction — Evaluation
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config

logger = logging.getLogger(__name__)


def evaluate_all(best_knn, baselines, X_train, X_test, y_train, y_test, knn_cv_score):
    """Evaluate tuned KNN vs all baselines on test set."""
    logger.info("=" * 60)
    logger.info("EVALUATION ON HELD-OUT TEST SET")
    logger.info("=" * 60)
    
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    
    all_models = {"KNN (Tuned GridSearch)": best_knn}
    all_models.update(baselines)
    
    results = []
    for name, model in all_models.items():
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = np.nan
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1 (Weighted)": f1,
            "Precision": prec,
            "Recall": rec,
            "AUC-ROC": auc,
            "CV F1 Mean": cv_scores.mean(),
            "CV F1 Std": cv_scores.std()
        })
        
        logger.info(f"\n  {name}")
        logger.info(f"    Accuracy={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f} | CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    results_df = pd.DataFrame(results).sort_values("F1 (Weighted)", ascending=False).reset_index(drop=True)
    
    logger.info(f"\n{'='*60}")
    logger.info("RANKINGS")
    logger.info(f"{'='*60}")
    for i, row in results_df.iterrows():
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        logger.info(f"  {medal} {row['Model']:30s} | F1={row['F1 (Weighted)']:.4f} | AUC={row['AUC-ROC']:.4f}")
    
    return results_df, all_models


def plot_confusion_matrices(all_models, X_test, y_test):
    """Confusion matrices for KNN (tuned) vs baselines."""
    n_models = len(all_models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    for idx, (name, model) in enumerate(all_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES)
        disp.plot(ax=axes[idx], cmap=plt.cm.Blues, colorbar=False, values_format='d')
        axes[idx].set_title(name, fontsize=11, fontweight='bold')
    
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_confusion_matrices.png")


def plot_model_comparison(results_df):
    """Bar chart comparing all models."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = results_df["Model"]
    x = np.arange(len(models))
    width = 0.2
    
    metrics = {"Accuracy": "#4FC3F7", "F1 (Weighted)": "#FF7043", "AUC-ROC": "#66BB6A"}
    
    for i, (metric, color) in enumerate(metrics.items()):
        vals = results_df[metric].values
        vals = np.nan_to_num(vals, nan=0)
        ax.barh(x + i*width, vals, width, label=metric, color=color, alpha=0.85, edgecolor='white')
    
    ax.set_yticks(x + width)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Tuned KNN vs Baselines", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1.05)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Highlight KNN row
    knn_idx = results_df[results_df["Model"].str.contains("Tuned")].index
    if len(knn_idx) > 0:
        y_pos = knn_idx[0]
        ax.axhspan(y_pos - 0.3, y_pos + 0.5, color='#FFD54F', alpha=0.15, zorder=0)
    
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_model_comparison.png")


def plot_roc_curves(all_models, X_test, y_test):
    """ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_models)))
    
    for (name, model), color in zip(all_models.items(), colors):
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            lw = 3 if "Tuned" in name else 1.5
            ax.plot(fpr, tpr, color=color, linewidth=lw, label=f"{name} (AUC={auc:.3f})")
        except:
            continue
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/07_roc_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 07_roc_curves.png")


def error_analysis(best_knn, X_test, y_test, feature_names):
    """Analyze misclassifications from the best KNN."""
    logger.info("=" * 60)
    logger.info("ERROR ANALYSIS — Tuned KNN")
    logger.info("=" * 60)
    
    y_pred = best_knn.predict(X_test)
    errors = y_pred != y_test
    n_errors = errors.sum()
    
    logger.info(f"  Misclassifications: {n_errors}/{len(y_test)} ({n_errors/len(y_test)*100:.1f}%)")
    
    if n_errors == 0:
        logger.info("  Perfect classification!")
        return
    
    # False positives vs false negatives
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    logger.info(f"  False Positives (predicted CKD, actual healthy): {fp}")
    logger.info(f"  False Negatives (missed CKD cases):             {fn}")
    logger.info(f"  ⚠️  In medicine, FN is more dangerous — missed disease!")
    
    # Feature comparison
    if n_errors > 0:
        logger.info(f"\n  Feature comparison (mean, scaled values):")
        correct_features = X_test[~errors]
        error_features = X_test[errors]
        
        for i, feat in enumerate(feature_names[:min(len(feature_names), X_test.shape[1])]):
            c_mean = correct_features[:, i].mean()
            e_mean = error_features[:, i].mean()
            logger.info(f"    {feat:25s} | Correct={c_mean:+.3f} | Error={e_mean:+.3f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, digits=4)
    logger.info(f"\n  Classification Report:\n{report}")


def save_results(results_df):
    """Save final results."""
    csv_path = f"{config.OUTPUT_DIR}/day06_results.csv"
    results_df.to_csv(csv_path, index=False, float_format='%.4f')
    
    report_path = f"{config.OUTPUT_DIR}/day06_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  DAY 6: KIDNEY DISEASE PREDICTION — FINAL REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("OBJECTIVE: Predict Chronic Kidney Disease using KNN with\n")
        f.write("           exhaustive GridSearchCV hyperparameter tuning.\n\n")
        f.write("DATASET: UCI CKD (400 samples, 24 features, ~35% missing)\n\n")
        f.write("-" * 70 + "\n")
        f.write("RESULTS\n")
        f.write("-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n")
        best = results_df.iloc[0]
        f.write(f"BEST MODEL: {best['Model']}\n")
        f.write(f"  F1={best['F1 (Weighted)']:.4f} | Acc={best['Accuracy']:.4f} | AUC={best['AUC-ROC']:.4f}\n\n")
        f.write("-" * 70 + "\n")
        f.write("KEY TAKEAWAYS\n")
        f.write("-" * 70 + "\n\n")
        f.write("1. GridSearchCV systematically tests all hyperparameter combos\n")
        f.write("2. The 'K' in KNN controls bias-variance tradeoff\n")
        f.write("3. Distance-weighted voting often beats uniform voting\n")
        f.write("4. Manhattan distance can be more robust than Euclidean\n")
        f.write("5. Missing value imputation strategy matters for KNN\n")
        f.write("6. Feature scaling is CRITICAL for distance-based models\n")
    
    logger.info(f"Results: {csv_path}")
    logger.info(f"Report: {report_path}")
