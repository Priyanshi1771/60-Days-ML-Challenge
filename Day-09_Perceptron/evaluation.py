"""
=============================================================================
 Day 9: Hepatitis Diagnosis — Evaluation (Optimized)
=============================================================================
 FOCUS: ROC Curve Analysis Deep-Dive
   1. Standard ROC curves for all models
   2. Threshold vs Metrics plot (how F1/Precision/Recall change)
   3. Optimal threshold selection (Youden's J statistic)
   4. Confidence intervals via bootstrapping
   5. Clinical decision analysis
=============================================================================
"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve, auc)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import config

logger = logging.getLogger(__name__)


def evaluate_all(calibrated, baselines, X_train, X_test, y_train, y_test):
    """Evaluate all models with AUC as primary metric."""
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)

    all_models = {"Perceptron (Calibrated)": calibrated}
    all_models.update(baselines)
    results = []

    for name, model in all_models.items():
        y_pred = model.predict(X_test)
        y_proba = _get_proba(model, X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        auc_score = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        results.append({"Model": name, "Accuracy": acc, "F1 (Weighted)": f1,
                         "Precision": prec, "Recall": rec, "AUC-ROC": auc_score})
        logger.info(f"  {name:28s} | Acc={acc:.4f} | F1={f1:.4f} | AUC={auc_score:.4f}")

    results_df = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)

    logger.info(f"\n{'='*60}\nRANKINGS (by AUC-ROC — Day 9 focus metric)\n{'='*60}")
    for i, row in results_df.iterrows():
        medal = ["🥇","🥈","🥉"][i] if i < 3 else "  "
        logger.info(f"  {medal} {row['Model']:28s} | AUC={row['AUC-ROC']:.4f} | F1={row['F1 (Weighted)']:.4f}")

    return results_df, all_models


def _get_proba(model, X):
    """Safely extract positive-class probabilities."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        return model.decision_function(X)
    return None


def plot_roc_deep_analysis(all_models, X_test, y_test):
    """
    THE ROC DEEP-DIVE — Core learning for Day 9.
    
    ROC CURVE EXPLAINED:
    - X-axis: False Positive Rate (FPR) = FP / (FP + TN)
      → "Of all healthy patients, what fraction did we falsely alarm?"
    - Y-axis: True Positive Rate (TPR) = TP / (TP + FN) = Recall
      → "Of all sick patients, what fraction did we correctly catch?"
    - Each point = one threshold value
    - AUC = Area Under Curve (1.0 = perfect, 0.5 = random)
    
    WHY AUC MATTERS:
    - Threshold-independent: evaluates model across ALL possible cutoffs
    - Works well for imbalanced data (unlike accuracy)
    - Clinically: separates "model quality" from "decision threshold"
    """
    logger.info("=" * 60)
    logger.info("ROC CURVE DEEP ANALYSIS")
    logger.info("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    colors = ['#EF5350', '#4FC3F7', '#66BB6A', '#FFB74D', '#AB47BC']

    # ─── (0,0): Standard ROC Curves ─────────────────────────────────
    ax = axes[0, 0]
    for idx, (name, model) in enumerate(all_models.items()):
        y_proba = _get_proba(model, X_test)
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        lw = 3 if "Perceptron" in name else 1.5
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=lw,
                label=f"{name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.500)')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    ax.set_title("📈 ROC Curves — All Models", fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

    # ─── (0,1): Threshold vs Metrics (Perceptron) ───────────────────
    ax = axes[0, 1]
    y_proba = _get_proba(all_models.get("Perceptron (Calibrated)",
                         list(all_models.values())[0]), X_test)
    if y_proba is not None:
        thresholds = np.linspace(*config.THRESHOLD_RANGE)
        f1s, precs, recs, accs = [], [], [], []
        for t in thresholds:
            y_t = (y_proba >= t).astype(int)
            if len(np.unique(y_t)) < 2:
                f1s.append(0); precs.append(0); recs.append(0); accs.append(0)
                continue
            f1s.append(f1_score(y_test, y_t, average='weighted'))
            precs.append(precision_score(y_test, y_t, average='weighted', zero_division=0))
            recs.append(recall_score(y_test, y_t, average='weighted'))
            accs.append(accuracy_score(y_test, y_t))

        ax.plot(thresholds, f1s, '-', color='#EF5350', lw=2.5, label='F1 (Weighted)')
        ax.plot(thresholds, precs, '--', color='#4FC3F7', lw=2, label='Precision')
        ax.plot(thresholds, recs, '--', color='#66BB6A', lw=2, label='Recall')
        ax.plot(thresholds, accs, ':', color='#FFB74D', lw=2, label='Accuracy')

        # Mark optimal threshold (max F1)
        best_t_idx = np.argmax(f1s)
        best_t = thresholds[best_t_idx]
        ax.axvline(x=best_t, color='#AB47BC', linestyle='-.', lw=2, alpha=0.7,
                   label=f'Best threshold={best_t:.2f}')
        ax.scatter([best_t], [f1s[best_t_idx]], s=100, color='#AB47BC', zorder=5, marker='*')

        # Mark default 0.5 threshold
        ax.axvline(x=0.5, color='gray', linestyle='--', lw=1, alpha=0.5, label='Default (0.5)')

        logger.info(f"  Optimal threshold: {best_t:.3f} (F1={f1s[best_t_idx]:.4f})")
        logger.info(f"  Default 0.5: F1={f1s[np.argmin(np.abs(thresholds-0.5))]:.4f}")

    ax.set_xlabel("Classification Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("🎯 Threshold vs Metrics — Where to Cut?\n(★ = optimal threshold for F1)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

    # ─── (1,0): Youden's J Statistic ────────────────────────────────
    ax = axes[1, 0]
    if y_proba is not None:
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
        j_scores = tpr - fpr  # Youden's J = Sensitivity - (1 - Specificity) = TPR - FPR
        best_j_idx = np.argmax(j_scores)
        best_j_threshold = roc_thresholds[best_j_idx]

        ax.plot(roc_thresholds, tpr, color='#66BB6A', lw=2, label='TPR (Sensitivity)')
        ax.plot(roc_thresholds, fpr, color='#EF5350', lw=2, label='FPR (1 - Specificity)')
        ax.plot(roc_thresholds, j_scores, color='#AB47BC', lw=2.5, label="Youden's J = TPR - FPR")
        ax.axvline(x=best_j_threshold, color='#FFB74D', linestyle='-.', lw=2,
                   label=f'Optimal (J={j_scores[best_j_idx]:.3f}, t={best_j_threshold:.3f})')

        logger.info(f"  Youden's J optimal: threshold={best_j_threshold:.3f}, J={j_scores[best_j_idx]:.3f}")

    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Rate / J Score", fontsize=11)
    ax.set_title("📊 Youden's J Statistic — Optimal Operating Point\n"
                 "(Maximizes distance between TPR and FPR)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 1)
    ax.spines[['top', 'right']].set_visible(False)

    # ─── (1,1): AUC Bootstrap Confidence Interval ────────────────────
    ax = axes[1, 1]
    if y_proba is not None:
        rng = np.random.RandomState(config.RANDOM_SEED)
        n_boot = 500
        boot_aucs = np.empty(n_boot, dtype=np.float32)
        n_test = len(y_test)

        for b in range(n_boot):
            idx = rng.randint(0, n_test, n_test)
            if len(np.unique(y_test[idx])) < 2:
                boot_aucs[b] = np.nan
                continue
            boot_aucs[b] = roc_auc_score(y_test[idx], y_proba[idx])

        boot_aucs = boot_aucs[~np.isnan(boot_aucs)]
        ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
        mean_auc = np.mean(boot_aucs)

        ax.hist(boot_aucs, bins=30, color='#4FC3F7', alpha=0.7, edgecolor='white')
        ax.axvline(x=mean_auc, color='#EF5350', lw=2.5, label=f'Mean AUC = {mean_auc:.3f}')
        ax.axvline(x=ci_low, color='#FFB74D', lw=2, linestyle='--', label=f'95% CI: [{ci_low:.3f}, {ci_high:.3f}]')
        ax.axvline(x=ci_high, color='#FFB74D', lw=2, linestyle='--')
        ax.axvspan(ci_low, ci_high, alpha=0.15, color='#FFB74D')

        logger.info(f"  Bootstrap AUC: {mean_auc:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")

    ax.set_xlabel("AUC-ROC", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("🎲 AUC Confidence Interval (500 bootstraps)\n"
                 "How stable is our AUC estimate?", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_roc_deep_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_roc_deep_analysis.png")


def plot_confusion_matrices(all_models, X_test, y_test):
    n = len(all_models)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes_flat = np.array(axes).ravel() if n > 1 else [axes]

    for idx, (name, model) in enumerate(all_models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES).plot(
            ax=axes_flat[idx], cmap=plt.cm.OrRd, colorbar=False, values_format='d')
        axes_flat[idx].set_title(name, fontsize=9, fontweight='bold')
    for j in range(idx+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("🦠 Confusion Matrices — Hepatitis Diagnosis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_confusion_matrices.png")


def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(12, 7))
    metrics = {"AUC-ROC": "#EF5350", "F1 (Weighted)": "#4FC3F7", "Accuracy": "#66BB6A"}
    x = np.arange(len(results_df))
    w = 0.22
    for i, (m, c) in enumerate(metrics.items()):
        vals = np.nan_to_num(results_df[m].values)
        ax.barh(x + i*w, vals, w, label=m, color=c, alpha=0.85, edgecolor='white')
    ax.set_yticks(x + w)
    ax.set_yticklabels(results_df["Model"], fontsize=10)
    ax.set_xlabel("Score"); ax.set_xlim(0, 1.05)
    ax.set_title("🦠 Model Comparison — Hepatitis Diagnosis", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right'); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_model_comparison.png")


def error_analysis(model, X_test, y_test):
    logger.info("=" * 60)
    logger.info("ERROR ANALYSIS — Perceptron (Calibrated)")
    logger.info("=" * 60)
    y_pred = model.predict(X_test)
    fp = ((y_pred==1)&(y_test==0)).sum()
    fn = ((y_pred==0)&(y_test==1)).sum()
    tp = ((y_pred==1)&(y_test==1)).sum()
    tn = ((y_pred==0)&(y_test==0)).sum()
    logger.info(f"  TP={tp} | TN={tn} | FP={fp} | FN={fn}")
    logger.info(f"  ⚠️  FN (predicted LIVE but DIED): {fn} — catastrophic misses!")
    logger.info(f"  FP (predicted DIE but LIVED): {fp} — unnecessary intervention")
    report = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, digits=4)
    logger.info(f"\n{report}")


def save_results(results_df):
    results_df.to_csv(f"{config.OUTPUT_DIR}/day09_results.csv", index=False, float_format='%.4f')
    with open(f"{config.OUTPUT_DIR}/day09_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 9: HEPATITIS DIAGNOSIS — FINAL REPORT\n" + "=" * 70 + "\n\n")
        f.write("OBJECTIVE: Diagnose hepatitis outcome using Perceptron\n")
        f.write("FOCUS: ROC curve analysis + threshold optimization\n")
        f.write("DATASET: UCI Hepatitis (155 samples, 19 features, ~20% Die)\n\n")
        f.write("-" * 70 + "\nRESULTS\n" + "-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "-" * 70 + "\nKEY TAKEAWAYS\n" + "-" * 70 + "\n\n")
        f.write("1. ROC curves evaluate models ACROSS ALL thresholds — not just 0.5\n")
        f.write("2. AUC is threshold-independent: separates model quality from cutoff choice\n")
        f.write("3. Youden's J finds the optimal operating point on the ROC curve\n")
        f.write("4. Perceptron is limited to linear boundaries — Day 10 fixes this with DL\n")
        f.write("5. Bootstrap CI shows how stable the AUC estimate is on small datasets\n")
        f.write("6. In hepatitis: FN (miss a death) > FP (false alarm) — optimize recall\n")
    logger.info("Results + report saved")
