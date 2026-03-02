"""
Day 2 - Breast Cancer Diagnosis
Evaluation: Metrics, Confusion Matrices, ROC, Feature Importance, Model Comparison.

WHAT'S NEW vs Day 1:
---------------------
1. Multi-model comparison (Unpruned vs Pre-pruned vs CCP-pruned)
2. Tree-specific feature importance (Gini importance vs Permutation importance)
3. Decision boundary intuition through feature importance
4. Side-by-side confusion matrices for easy comparison

TREE FEATURE IMPORTANCE — A WARNING:
--------------------------------------
Decision Trees have a built-in feature importance metric: "Gini importance"
(also called Mean Decrease in Impurity / MDI). It measures how much each
feature reduces impurity across all nodes where it's used for splitting.

PROBLEM: Gini importance is BIASED toward high-cardinality features.
A feature with many unique values has more possible split points, so it
gets more chances to reduce impurity, even if it's not truly important.

SOLUTION: Always compare Gini importance with Permutation importance
(which shuffles each feature and measures the actual performance drop).
If they disagree, trust permutation importance.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

from config import PLOT_DIR, OUTPUT_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)


# ============================================================
# 1. SINGLE MODEL EVALUATION
# ============================================================
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Compute comprehensive metrics for a single model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # P(benign)

    # NOTE: target encoding is 0=malignant, 1=benign
    # So "positive class" (1) is benign.
    # For CLINICAL relevance, we care about catching malignant (0).
    # We'll report metrics for BOTH classes.

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_benign': precision_score(y_test, y_pred, pos_label=1),
        'recall_benign': recall_score(y_test, y_pred, pos_label=1),
        'precision_malignant': precision_score(y_test, y_pred, pos_label=0),
        'recall_malignant': recall_score(y_test, y_pred, pos_label=0),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'auc_roc': roc_auc_score(y_test, y_proba),
    }

    logger.info(f"\n{'=' * 60}")
    logger.info(f"EVALUATION: {model_name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Accuracy:               {metrics['accuracy']:.4f}")
    logger.info(f"  Weighted F1:            {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:                {metrics['auc_roc']:.4f}")
    logger.info(f"  --- Malignant (cancer) detection ---")
    logger.info(f"  Precision (malignant):  {metrics['precision_malignant']:.4f}")
    logger.info(f"  Recall (malignant):     {metrics['recall_malignant']:.4f}  ← CRITICAL (missed cancers)")
    logger.info(f"  --- Benign detection ---")
    logger.info(f"  Precision (benign):     {metrics['precision_benign']:.4f}")
    logger.info(f"  Recall (benign):        {metrics['recall_benign']:.4f}")

    report = classification_report(y_test, y_pred,
                                   target_names=['Malignant', 'Benign'])
    logger.info(f"\n{report}")

    return metrics, y_pred, y_proba


# ============================================================
# 2. MULTI-MODEL COMPARISON
# ============================================================
def compare_models(all_results):
    """
    Create a comparison table and plot for all models.

    WHY COMPARE MULTIPLE APPROACHES?
    ----------------------------------
    Seeing the unpruned, pre-pruned, and CCP-pruned trees side by side
    teaches you:
    1. How much overfitting affects the unpruned tree
    2. Whether pre-pruning or CCP finds a better bias-variance tradeoff
    3. Which approach is simpler and more interpretable

    In practice, you'd also compare trees against other model families
    (Logistic Regression, SVM, etc.) to decide the final model.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    model_names = list(all_results.keys())
    metrics_to_plot = ['accuracy', 'f1', 'auc_roc', 'recall_malignant']
    metric_labels = ['Accuracy', 'Weighted F1', 'AUC-ROC', 'Recall\n(Malignant)']

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 5))

    colors = ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726']

    for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
        values = [all_results[m]['metrics'][metric] for m in model_names]
        bars = ax.bar(range(len(model_names)), values, color=colors[:len(model_names)])

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Model Comparison: Pruning Strategy Impact',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(PLOT_DIR, "model_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"\nModel comparison plot saved to: {filepath}")

    # Log comparison table
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{'MODEL COMPARISON TABLE':^80}")
    logger.info(f"{'=' * 80}")
    header = f"{'Metric':<25}"
    for name in model_names:
        header += f"│ {name:>16} "
    logger.info(header)
    logger.info("-" * 80)

    for metric in ['accuracy', 'f1', 'auc_roc', 'recall_malignant', 'precision_malignant']:
        row = f"  {metric:<23}"
        values = [all_results[m]['metrics'][metric] for m in model_names]
        best_val = max(values)
        for val in values:
            marker = " ★" if val == best_val else "  "
            row += f"│   {val:.4f}{marker}       "
        logger.info(row)


# ============================================================
# 3. CONFUSION MATRICES (Side by Side)
# ============================================================
def plot_confusion_matrices(all_results, y_test):
    """Plot confusion matrices for all models side by side."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, all_results.items()):
        cm = confusion_matrix(y_test, data['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Malignant', 'Benign'],
                    yticklabels=['Malignant', 'Benign'],
                    annot_kws={"size": 14}, ax=ax)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(f'{name}\nAcc={data["metrics"]["accuracy"]:.3f}', fontsize=11)

        # Highlight false negatives
        tn, fp, fn, tp = cm.ravel()
        if fn > 0:
            ax.text(1.5, -0.3, f'⚠️ {fn} missed cancer(s)',
                    ha='center', fontsize=9, color='red', fontweight='bold',
                    transform=ax.transData)

    plt.suptitle('Confusion Matrices — Model Comparison\n'
                 '(Focus: minimizing missed cancers = bottom-left cell)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(PLOT_DIR, "confusion_matrices_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrices saved to: {filepath}")


# ============================================================
# 4. ROC CURVES (All models overlaid)
# ============================================================
def plot_roc_curves(all_results, y_test):
    """Plot ROC curves for all models on a single plot."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 7))
    colors = ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726']

    for (name, data), color in zip(all_results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, data['y_proba'])
        auc = data['metrics']['auc_roc']
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.500)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — Pruning Strategy Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(PLOT_DIR, "roc_curves_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curves saved to: {filepath}")


# ============================================================
# 5. FEATURE IMPORTANCE (Gini vs Permutation)
# ============================================================
def analyze_feature_importance(model, X_test, y_test, feature_names, model_name="Model"):
    """
    Compare tree's built-in Gini importance with permutation importance.

    GINI IMPORTANCE (MDI — Mean Decrease in Impurity):
    For each feature, sum the total reduction in Gini impurity across
    all nodes where that feature is used for splitting, weighted by
    the number of samples reaching each node.
    + Fast (computed during training)
    - Biased toward features with many unique values
    - Doesn't account for feature correlations

    PERMUTATION IMPORTANCE:
    For each feature, randomly shuffle its values and measure how much
    the model's performance drops. Bigger drop = more important.
    + Unbiased
    + Captures actual predictive impact
    - Slower (requires re-evaluation for each feature)
    - Correlated features split importance between them
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- Gini Importance ---
    gini_imp = model.feature_importances_
    gini_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': gini_imp
    }).sort_values('Importance', ascending=True).tail(15)

    axes[0].barh(gini_df['Feature'], gini_df['Importance'], color='#42A5F5')
    axes[0].set_xlabel('Gini Importance (MDI)', fontsize=11)
    axes[0].set_title(f'Gini Feature Importance\n(Built-in, biased toward high-cardinality)',
                      fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='x')

    # --- Permutation Importance ---
    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=30, random_state=RANDOM_SEED,
        scoring='f1_weighted'
    )

    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_result.importances_mean,
        'Std': perm_result.importances_std
    }).sort_values('Importance', ascending=True).tail(15)

    axes[1].barh(perm_df['Feature'], perm_df['Importance'],
                 xerr=perm_df['Std'], color='#66BB6A', capsize=3)
    axes[1].set_xlabel('Mean F1 Decrease', fontsize=11)
    axes[1].set_title('Permutation Importance\n(Unbiased, model-agnostic)',
                      fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.suptitle(f'Feature Importance Analysis — {model_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(PLOT_DIR, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance plot saved to: {filepath}")

    # Log top features
    logger.info(f"\nTop 5 Features by Gini Importance ({model_name}):")
    top_gini = pd.DataFrame({'Feature': feature_names, 'Importance': gini_imp})
    top_gini = top_gini.sort_values('Importance', ascending=False).head(5)
    for _, row in top_gini.iterrows():
        logger.info(f"  {row['Feature']:>30}: {row['Importance']:.4f}")

    logger.info(f"\nTop 5 Features by Permutation Importance ({model_name}):")
    top_perm = perm_df.sort_values('Importance', ascending=False).head(5)
    for _, row in top_perm.iterrows():
        logger.info(f"  {row['Feature']:>30}: {row['Importance']:.4f} ± {row['Std']:.4f}")


# ============================================================
# 6. ERROR ANALYSIS
# ============================================================
def error_analysis(model, X_test, y_test, y_pred, feature_names):
    """
    Analyze misclassified samples with clinical context.

    For breast cancer, the critical errors are:
    - FALSE NEGATIVES: Malignant tumor predicted as benign → patient doesn't get treatment
    - FALSE POSITIVES: Benign tumor predicted as malignant → unnecessary biopsy/surgery

    We examine what features characterize the misclassified samples.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("ERROR ANALYSIS")
    logger.info(f"{'=' * 60}")

    error_df = pd.DataFrame(X_test.values, columns=feature_names)
    error_df['true_label'] = y_test.values
    error_df['predicted'] = y_pred
    error_df['correct'] = error_df['true_label'] == error_df['predicted']
    error_df['true_name'] = error_df['true_label'].map({0: 'malignant', 1: 'benign'})
    error_df['pred_name'] = error_df['predicted'].map({0: 'malignant', 1: 'benign'})

    # Count each type
    total = len(error_df)
    correct = error_df['correct'].sum()
    fn = ((error_df['true_label'] == 0) & (error_df['predicted'] == 1)).sum()
    fp = ((error_df['true_label'] == 1) & (error_df['predicted'] == 0)).sum()

    logger.info(f"Total test samples:  {total}")
    logger.info(f"Correct predictions: {correct} ({correct/total*100:.1f}%)")
    logger.info(f"False Negatives:     {fn} (MISSED CANCERS — critical!)")
    logger.info(f"False Positives:     {fp} (unnecessary procedures)")

    # Analyze false negatives in detail
    fn_cases = error_df[(error_df['true_label'] == 0) & (error_df['predicted'] == 1)]
    if len(fn_cases) > 0:
        logger.info(f"\nFalse Negative Analysis ({len(fn_cases)} missed cancers):")
        # Compare mean feature values
        malignant_correct = error_df[(error_df['true_label'] == 0) & (error_df['correct'])]

        key_features = ['worst radius', 'worst concave points', 'mean concave points',
                        'worst perimeter', 'mean radius']
        for feat in key_features:
            if feat in feature_names:
                fn_mean = fn_cases[feat].mean()
                correct_mean = malignant_correct[feat].mean() if len(malignant_correct) > 0 else 0
                logger.info(f"  {feat:>25}: FN={fn_mean:.3f} vs Correct Malignant={correct_mean:.3f}")

        logger.info("  (FN cases likely have feature values closer to benign range)")
    else:
        logger.info("\nNo false negatives! All malignant cases were correctly detected.")

    return error_df
