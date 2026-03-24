"""
=============================================================================
 Day 7: Stroke Risk Prediction — Evaluation
=============================================================================
"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve, average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import config

logger = logging.getLogger(__name__)


def evaluate_all(best_model, baselines, X_train, X_test, y_train, y_test, best_cv_f1):
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    all_models = {"XGBoost (Tuned)": best_model}
    all_models.update(baselines)
    results = []

    for name, model in all_models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        try:
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
            auc = roc_auc_score(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
        except:
            auc, ap = np.nan, np.nan

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        results.append({"Model": name, "Accuracy": acc, "F1": f1, "Precision": prec,
                         "Recall": rec, "AUC-ROC": auc, "Avg Precision": ap,
                         "CV F1 Mean": cv_scores.mean(), "CV F1 Std": cv_scores.std()})

        logger.info(f"\n  {name}")
        logger.info(f"    Acc={acc:.4f} | F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | AUC={auc:.4f} | AP={ap:.4f}")

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)

    logger.info(f"\n{'='*60}\nRANKINGS (by F1 — critical for imbalanced data)\n{'='*60}")
    for i, row in results_df.iterrows():
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        logger.info(f"  {medal} {row['Model']:25s} | F1={row['F1']:.4f} | Recall={row['Recall']:.4f}")

    return results_df, all_models


def plot_confusion_matrices(all_models, X_test, y_test):
    n = len(all_models)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1: axes = [axes]

    for idx, (name, model) in enumerate(all_models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES).plot(
            ax=axes[idx], cmap=plt.cm.Blues, colorbar=False, values_format='d')
        axes[idx].set_title(name, fontsize=10, fontweight='bold')

    fig.suptitle("🧠 Confusion Matrices — Stroke Prediction", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_confusion_matrices.png")


def plot_roc_and_pr_curves(all_models, X_test, y_test):
    """Plot ROC AND Precision-Recall curves (PR is better for imbalanced data)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_models)))

    for (name, model), color in zip(all_models.items(), colors):
        try:
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            lw = 3 if "XGBoost" in name else 1.5
            axes[0].plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC={auc:.3f})")
            # PR
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            axes[1].plot(rec, prec, color=color, lw=lw, label=f"{name} (AP={ap:.3f})")
        except: continue

    axes[0].plot([0,1], [0,1], 'k--', alpha=0.3)
    axes[0].set_title("⚡ ROC Curves", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.2)
    axes[0].spines[['top','right']].set_visible(False)

    baseline = y_test.mean()
    axes[1].axhline(y=baseline, color='k', linestyle='--', alpha=0.3, label=f'Baseline ({baseline:.3f})')
    axes[1].set_title("🩺 Precision-Recall Curves (better for imbalanced)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.2)
    axes[1].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_roc_pr_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_roc_pr_curves.png")


def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(12, 7))
    metrics = {"F1": "#EF5350", "Recall": "#4FC3F7", "Precision": "#66BB6A", "AUC-ROC": "#FFB74D"}
    x = np.arange(len(results_df))
    w = 0.18
    for i, (m, c) in enumerate(metrics.items()):
        vals = results_df[m].values
        ax.barh(x + i*w, np.nan_to_num(vals), w, label=m, color=c, alpha=0.85, edgecolor='white')
    ax.set_yticks(x + 1.5*w)
    ax.set_yticklabels(results_df["Model"], fontsize=10)
    ax.set_xlabel("Score"); ax.set_title("🧠 Model Comparison — Stroke Prediction", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right'); ax.set_xlim(0, 1.05); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/07_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 07_model_comparison.png")


def error_analysis(model, X_test, y_test, feature_names):
    logger.info("=" * 60)
    logger.info("ERROR ANALYSIS — XGBoost")
    logger.info("=" * 60)
    y_pred = model.predict(X_test)
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    tn = ((y_pred == 0) & (y_test == 0)).sum()
    logger.info(f"  TP={tp} | TN={tn} | FP={fp} | FN={fn}")
    logger.info(f"  ⚠️  Missed strokes (FN): {fn} — these patients would go undiagnosed!")
    logger.info(f"  False alarms (FP): {fp} — unnecessary follow-ups")
    if fn > 0:
        logger.info(f"  🚨 FN rate: {fn/(fn+tp)*100:.1f}% of stroke cases MISSED")
    report = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, digits=4)
    logger.info(f"\n{report}")


def save_results(results_df):
    csv_path = f"{config.OUTPUT_DIR}/day07_results.csv"
    results_df.to_csv(csv_path, index=False, float_format='%.4f')

    report_path = f"{config.OUTPUT_DIR}/day07_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  DAY 7: STROKE RISK PREDICTION — FINAL REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("OBJECTIVE: Predict stroke risk with XGBoost + SHAP interpretability\n")
        f.write("DATASET: Kaggle Stroke (5110 samples, ~5% positive rate)\n")
        f.write("KEY CHALLENGE: Extreme class imbalance (95:5)\n\n")
        f.write("-" * 70 + "\nRESULTS\n" + "-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n")
        best = results_df.iloc[0]
        f.write(f"BEST MODEL: {best['Model']}\n")
        f.write(f"  F1={best['F1']:.4f} | Recall={best['Recall']:.4f} | AUC={best['AUC-ROC']:.4f}\n\n")
        f.write("-" * 70 + "\nKEY TAKEAWAYS\n" + "-" * 70 + "\n\n")
        f.write("1. SHAP reveals WHY the model predicts — not just WHAT it predicts\n")
        f.write("2. Age is the strongest stroke predictor (SHAP confirms clinical knowledge)\n")
        f.write("3. scale_pos_weight is critical for handling 95:5 imbalance in XGBoost\n")
        f.write("4. F1 and PR-AUC are better metrics than accuracy for imbalanced data\n")
        f.write("5. Precision-Recall curves reveal more than ROC for rare events\n")
        f.write("6. In stroke prediction, Recall > Precision (missing a stroke is worse)\n")
    logger.info(f"Results: {csv_path}")
    logger.info(f"Report: {report_path}")
