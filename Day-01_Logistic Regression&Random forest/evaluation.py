"""
Day 1 - Heart Disease Prediction
Evaluation: Metrics, Visualization, Error Analysis, Threshold Tuning.

WHAT THIS MODULE DOES:
----------------------
After training, you need to rigorously evaluate your model. This module provides:
  1. Standard metrics: Accuracy, Precision, Recall, F1, AUC
  2. Confusion matrix visualization
  3. ROC curve plotting
  4. Feature importance analysis
  5. Threshold tuning (critical for medical applications!)
  6. Error analysis (understanding WHERE the model fails)

WHY SO MANY METRICS?
---------------------
In medicine, ACCURACY ALONE IS DANGEROUS. Here's why:

Imagine a disease with 95% healthy, 5% sick patients.
A model that ALWAYS predicts "healthy" gets 95% accuracy — but misses
every single sick patient! That's a useless (and dangerous) model.

That's why we need:
  - Precision: Of those predicted sick, how many truly are?
  - Recall: Of those truly sick, how many did we catch?
  - F1 Score: Harmonic mean of precision and recall (balances both)
  - AUC: Overall ranking quality across all thresholds

For heart disease: RECALL IS KING. Missing a sick patient (false negative)
could cost a life. A false alarm (false positive) just means extra tests.

HOW TO DEBUG:
-------------
  - Low recall? Model is too conservative → lower the threshold
  - Low precision? Too many false alarms → raise the threshold
  - AUC close to 0.5? Model is barely better than random → revisit features
  - AUC > 0.95 on small dataset? Suspicious → check for data leakage!
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance

from config import PLOT_DIR, OUTPUT_DIR, CLASSIFICATION_THRESHOLD

logger = logging.getLogger(__name__)


# ============================================================
# 1. COMPREHENSIVE METRICS
# ============================================================
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Compute all relevant classification metrics.

    Returns a dictionary of metrics for easy comparison between models.

    KEY INSIGHT - PROBABILITY vs HARD PREDICTION:
    ------------------------------------------------
    model.predict(X) returns hard labels (0 or 1) using threshold 0.5.
    model.predict_proba(X)[:, 1] returns probabilities.

    Hard predictions lose information! A patient with 0.49 probability and
    one with 0.01 probability are both labeled "healthy" by predict(),
    but clinically they're very different. Always use probabilities when possible.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"EVALUATION: {model_name}")
    logger.info(f"{'=' * 60}")

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'avg_precision': average_precision_score(y_test, y_proba)
    }

    # Log results
    logger.info(f"Accuracy:           {metrics['accuracy']:.4f}")
    logger.info(f"Precision:          {metrics['precision']:.4f}")
    logger.info(f"Recall:             {metrics['recall']:.4f}")
    logger.info(f"F1 Score:           {metrics['f1']:.4f}")
    logger.info(f"AUC-ROC:            {metrics['auc_roc']:.4f}")
    logger.info(f"Avg Precision (AP): {metrics['avg_precision']:.4f}")

    # Full classification report
    report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])
    logger.info(f"\nClassification Report:\n{report}")

    return metrics, y_pred, y_proba


# ============================================================
# 2. CONFUSION MATRIX VISUALIZATION
# ============================================================
def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    """
    Plot a clear, annotated confusion matrix.

    HOW TO READ A CONFUSION MATRIX:
    ---------------------------------
                        Predicted
                    Negative  |  Positive
    Actual  Negative   TN     |    FP      ← False Positive (Type I error)
            Positive   FN     |    TP      ← False Negative (Type II error)

    For heart disease:
    - TN: Correctly predicted healthy → patient goes home
    - TP: Correctly predicted disease → patient gets treatment
    - FP: Predicted disease but actually healthy → unnecessary tests (annoying but safe)
    - FN: Predicted healthy but actually diseased → DANGEROUS! Patient goes untreated!

    The FN cell is what we want to minimize in medical diagnosis.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))

    # Create annotated heatmap
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Disease', 'Disease'],
        yticklabels=['No Disease', 'Disease'],
        annot_kws={"size": 16}
    )

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)

    # Add interpretation text
    tn, fp, fn, tp = cm.ravel()
    plt.figtext(0.5, -0.05,
                f"TN={tn} | FP={fp} (false alarms) | FN={fn} (MISSED cases!) | TP={tp}",
                ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to: {filepath}")

    return cm


# ============================================================
# 3. ROC CURVE
# ============================================================
def plot_roc_curve(y_test, y_proba, model_name="Model"):
    """
    Plot the ROC (Receiver Operating Characteristic) curve.

    WHAT ROC TELLS YOU:
    --------------------
    The ROC curve plots True Positive Rate (Recall) vs False Positive Rate
    at every possible threshold from 0 to 1.

    - A perfect model hugs the top-left corner (AUC = 1.0)
    - A random model follows the diagonal (AUC = 0.5)
    - The closer AUC is to 1.0, the better your model separates classes

    WHY IS AUC USEFUL?
    -------------------
    AUC is THRESHOLD-INDEPENDENT. It tells you: "If I pick a random sick
    patient and a random healthy patient, what's the probability the model
    assigns a higher score to the sick one?" AUC = that probability.

    This means AUC evaluates the model's RANKING ability, not its calibration.
    A model could have AUC = 0.95 but terrible accuracy at threshold 0.5.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2196F3', lw=2, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random (AUC = 0.500)')
    plt.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')

    # Mark the default threshold point
    default_idx = np.argmin(np.abs(thresholds - CLASSIFICATION_THRESHOLD))
    plt.scatter(fpr[default_idx], tpr[default_idx], color='red', s=100, zorder=5,
                label=f'Threshold={CLASSIFICATION_THRESHOLD}')

    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, f"roc_curve_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved to: {filepath}")


# ============================================================
# 4. FEATURE IMPORTANCE ANALYSIS
# ============================================================
def analyze_feature_importance(model, X_test, y_test, feature_names, model_name="Logistic Regression"):
    """
    Analyze and visualize feature importance.

    TWO TYPES OF FEATURE IMPORTANCE:
    -----------------------------------
    1. COEFFICIENT-BASED (Logistic Regression specific):
       After scaling, the coefficient magnitude tells you how much each feature
       influences the prediction. Larger |coefficient| = more important.
       Sign tells direction: positive = increases disease risk.

    2. PERMUTATION IMPORTANCE (model-agnostic):
       Randomly shuffle one feature at a time and measure how much accuracy drops.
       If shuffling a feature causes a big drop → that feature is important.
       If shuffling has no effect → that feature is useless.

       WHY IS THIS BETTER THAN COEFFICIENTS?
       Permutation importance captures nonlinear effects and interactions,
       while coefficients only capture linear contributions.

    COMMON MISTAKE: Comparing raw coefficients of unscaled features.
    A coefficient of 0.5 for age (range 29-77) means something very different
    than 0.5 for fbs (range 0-1). ALWAYS scale features before interpreting
    coefficient magnitudes.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ----- Plot 1: Coefficient-based importance (LR only) -----
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).sort_values('Coefficient', key=abs, ascending=True)

        colors = ['#EF5350' if c > 0 else '#42A5F5' for c in coef_df['Coefficient']]
        axes[0].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
        axes[0].set_xlabel('Coefficient Value', fontsize=11)
        axes[0].set_title(f'Logistic Regression Coefficients\n(Red=↑ Disease Risk, Blue=↓ Risk)', fontsize=12)
        axes[0].axvline(x=0, color='black', linewidth=0.8)
        axes[0].grid(True, alpha=0.3, axis='x')

    # ----- Plot 2: Permutation Importance (model-agnostic) -----
    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=30,
        random_state=42,
        scoring='f1'
    )

    perm_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_result.importances_mean,
        'Std': perm_result.importances_std
    }).sort_values('Importance', ascending=True)

    axes[1].barh(perm_df['Feature'], perm_df['Importance'],
                 xerr=perm_df['Std'], color='#66BB6A', capsize=3)
    axes[1].set_xlabel('Mean F1 Decrease', fontsize=11)
    axes[1].set_title('Permutation Importance\n(How much F1 drops when feature is shuffled)', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.suptitle(f'Feature Importance Analysis - {model_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance plot saved to: {filepath}")

    # Log top features
    if hasattr(model, 'coef_'):
        top_features = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)

        logger.info("\nTop Features by |Coefficient|:")
        for _, row in top_features.head(5).iterrows():
            direction = "↑ risk" if row['Coefficient'] > 0 else "↓ risk"
            logger.info(f"  {row['Feature']:>12}: {row['Coefficient']:>+.4f} ({direction})")


# ============================================================
# 5. THRESHOLD TUNING (Critical for Medical Applications!)
# ============================================================
def find_optimal_threshold(y_test, y_proba, model_name="Model"):
    """
    Find the optimal classification threshold.

    WHY NOT ALWAYS USE 0.5?
    ------------------------
    The default threshold of 0.5 treats false positives and false negatives
    as equally bad. But in medicine, they're NOT equal:

    - False Negative (missed disease): Could cost a life → very expensive
    - False Positive (false alarm): Extra tests → mildly expensive

    By LOWERING the threshold (e.g., to 0.3), we classify more patients as
    "disease," catching more truly sick patients (higher recall) at the cost
    of more false alarms (lower precision).

    STRATEGIES FOR CHOOSING THRESHOLD:
    ------------------------------------
    1. Maximize F1 Score (balanced approach)
    2. Maximize Recall to ≥ 0.90 (clinical safety requirement)
    3. Youden's J statistic: max(TPR - FPR) — optimal balance point on ROC
    4. Cost-based: if you know the dollar cost of FP vs FN, minimize total cost

    For heart disease, we'll find: (a) best F1 threshold, (b) threshold for 90% recall.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Method 1: Precision-Recall curve to find best F1 threshold
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds_pr[best_f1_idx] if best_f1_idx < len(thresholds_pr) else 0.5

    # Method 2: Find threshold for >= 90% recall
    target_recall = 0.90
    valid_indices = np.where(recalls >= target_recall)[0]
    if len(valid_indices) > 0:
        # Among thresholds achieving >= 90% recall, pick the one with highest precision
        best_90_idx = valid_indices[np.argmax(precisions[valid_indices])]
        threshold_90_recall = thresholds_pr[best_90_idx] if best_90_idx < len(thresholds_pr) else 0.3
    else:
        threshold_90_recall = 0.3

    # Method 3: Youden's J statistic (ROC-based)
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    youden_threshold = thresholds_roc[best_j_idx]

    logger.info(f"\nThreshold Analysis:")
    logger.info(f"  Best F1 Threshold:        {best_f1_threshold:.3f}")
    logger.info(f"  90% Recall Threshold:     {threshold_90_recall:.3f}")
    logger.info(f"  Youden's J Threshold:     {youden_threshold:.3f}")

    # Visualize threshold effects
    test_thresholds = np.arange(0.1, 0.91, 0.05)
    accs, precs, recs, f1s = [], [], [], []

    for t in test_thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        accs.append(accuracy_score(y_test, y_pred_t))
        precs.append(precision_score(y_test, y_pred_t, zero_division=0))
        recs.append(recall_score(y_test, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

    plt.figure(figsize=(10, 6))
    plt.plot(test_thresholds, accs, label='Accuracy', marker='o', markersize=4)
    plt.plot(test_thresholds, precs, label='Precision', marker='s', markersize=4)
    plt.plot(test_thresholds, recs, label='Recall', marker='^', markersize=4)
    plt.plot(test_thresholds, f1s, label='F1 Score', marker='D', markersize=4, linewidth=2)

    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
    plt.axvline(x=best_f1_threshold, color='red', linestyle='--', alpha=0.7,
                label=f'Best F1 ({best_f1_threshold:.2f})')
    plt.axhline(y=0.90, color='green', linestyle=':', alpha=0.5, label='90% Recall Target')

    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Threshold vs Metrics - {model_name}', fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, f"threshold_tuning_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Threshold tuning plot saved to: {filepath}")

    return {
        'best_f1_threshold': best_f1_threshold,
        'threshold_90_recall': threshold_90_recall,
        'youden_threshold': youden_threshold
    }


# ============================================================
# 6. ERROR ANALYSIS
# ============================================================
def error_analysis(model, X_test, y_test, y_pred, y_proba, feature_names):
    """
    Understand WHERE and WHY the model fails.

    WHY ERROR ANALYSIS MATTERS:
    ----------------------------
    Aggregate metrics (accuracy, F1) tell you HOW WELL the model does overall.
    Error analysis tells you HOW and WHERE it fails — which is far more actionable.

    By examining misclassified patients, you might discover:
    - The model struggles with borderline cases (probabilities near 0.5)
    - Certain subgroups (e.g., young females with atypical chest pain) are poorly predicted
    - Specific feature patterns that consistently confuse the model

    This analysis directly informs your next steps:
    - If errors cluster in a subgroup → collect more data for that group
    - If errors are near the boundary → threshold tuning or ensemble methods
    - If errors have a pattern → engineer new features to capture that pattern
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("ERROR ANALYSIS")
    logger.info(f"{'=' * 60}")

    # Create a DataFrame for analysis
    error_df = pd.DataFrame(X_test, columns=feature_names)
    error_df['true_label'] = y_test.values
    error_df['predicted'] = y_pred
    error_df['probability'] = y_proba
    error_df['correct'] = (error_df['true_label'] == error_df['predicted'])
    error_df['error_type'] = 'Correct'
    error_df.loc[(error_df['true_label'] == 1) & (error_df['predicted'] == 0), 'error_type'] = 'False Negative'
    error_df.loc[(error_df['true_label'] == 0) & (error_df['predicted'] == 1), 'error_type'] = 'False Positive'

    # Count errors
    error_counts = error_df['error_type'].value_counts()
    logger.info(f"Error breakdown:\n{error_counts}")

    # Analyze false negatives (most dangerous in medical context)
    fn_cases = error_df[error_df['error_type'] == 'False Negative']
    if len(fn_cases) > 0:
        logger.info(f"\nFalse Negatives ({len(fn_cases)} cases) - MISSED DISEASE:")
        logger.info(f"  Average probability: {fn_cases['probability'].mean():.3f}")
        logger.info(f"  (These patients were close to the threshold but fell below it)")
        logger.info(f"  Feature averages vs overall test set:")
        for feat in feature_names[:5]:  # Top 5 features for brevity
            fn_mean = fn_cases[feat].mean()
            overall_mean = error_df[feat].mean()
            diff = fn_mean - overall_mean
            logger.info(f"    {feat:>12}: FN mean={fn_mean:.2f}, Overall={overall_mean:.2f}, Diff={diff:+.2f}")

    # Analyze false positives
    fp_cases = error_df[error_df['error_type'] == 'False Positive']
    if len(fp_cases) > 0:
        logger.info(f"\nFalse Positives ({len(fp_cases)} cases) - FALSE ALARMS:")
        logger.info(f"  Average probability: {fp_cases['probability'].mean():.3f}")

    # Confidence analysis
    logger.info(f"\nConfidence Distribution:")
    logger.info(f"  Correct predictions - avg confidence: {error_df[error_df['correct']]['probability'].apply(lambda p: max(p, 1-p)).mean():.3f}")
    if len(error_df[~error_df['correct']]) > 0:
        logger.info(f"  Wrong predictions   - avg confidence: {error_df[~error_df['correct']]['probability'].apply(lambda p: max(p, 1-p)).mean():.3f}")
        logger.info("  (If wrong predictions are also high-confidence, the model is confidently wrong → concerning!)")

    return error_df
