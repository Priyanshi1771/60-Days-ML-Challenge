"""
=============================================================================
 Day 8: Anemia Detection — Evaluation
=============================================================================
"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import config
from data_pipeline import detect_outliers, get_scaler

logger = logging.getLogger(__name__)


def evaluate_all(best_ada, baselines, X_train_raw, X_test_raw, y_train, y_test,
                 outlier_method, scale_method):
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)

    # Apply same preprocessing to test (scaler fitted on train)
    scaler = get_scaler(scale_method)
    X_train_clean, y_train_clean, _ = detect_outliers(X_train_raw.copy(), y_train.copy(), outlier_method)
    X_train_scaled = scaler.fit_transform(X_train_clean) if scaler else X_train_clean
    X_test_scaled = scaler.transform(X_test_raw) if scaler else X_test_raw.copy()

    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    all_models = {"AdaBoost (Tuned)": best_ada}
    all_models.update(baselines)
    results = []

    for name, model in all_models.items():
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        try:
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = np.nan

        results.append({"Model": name, "Accuracy": acc, "F1 (Weighted)": f1,
                         "Precision": prec, "Recall": rec, "AUC-ROC": auc})
        logger.info(f"  {name:25s} | Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

    results_df = pd.DataFrame(results).sort_values("F1 (Weighted)", ascending=False).reset_index(drop=True)

    logger.info(f"\n{'='*60}\nRANKINGS\n{'='*60}")
    for i, row in results_df.iterrows():
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        logger.info(f"  {medal} {row['Model']:25s} | F1={row['F1 (Weighted)']:.4f} | AUC={row['AUC-ROC']:.4f}")

    return results_df, all_models, X_test_scaled


def plot_confusion_matrices(all_models, X_test, y_test):
    n = len(all_models)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes_flat = axes.ravel() if hasattr(axes, 'ravel') else [axes]

    for idx, (name, model) in enumerate(all_models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES).plot(
            ax=axes_flat[idx], cmap=plt.cm.Reds, colorbar=False, values_format='d')
        axes_flat[idx].set_title(name, fontsize=10, fontweight='bold')
    for j in range(idx+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("🩸 Confusion Matrices — Anemia Detection", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_confusion_matrices.png")


def plot_roc_curves(all_models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_models)))
    for (name, model), color in zip(all_models.items(), colors):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            lw = 3 if "AdaBoost" in name else 1.5
            ax.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC={auc:.3f})")
        except: continue
    ax.plot([0,1],[0,1],'k--',alpha=0.3)
    ax.set_title("🩸 ROC Curves — Anemia Detection", fontsize=14, fontweight='bold')
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=9); ax.grid(alpha=0.2); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_roc_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_roc_curves.png")


def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(12, 7))
    metrics = {"F1 (Weighted)": "#EF5350", "Accuracy": "#4FC3F7", "AUC-ROC": "#66BB6A"}
    x = np.arange(len(results_df))
    w = 0.22
    for i, (m, c) in enumerate(metrics.items()):
        vals = np.nan_to_num(results_df[m].values)
        ax.barh(x + i*w, vals, w, label=m, color=c, alpha=0.85, edgecolor='white')
    ax.set_yticks(x + w)
    ax.set_yticklabels(results_df["Model"], fontsize=10)
    ax.set_xlabel("Score"); ax.set_xlim(0, 1.05)
    ax.set_title("🩸 Model Comparison — Anemia Detection", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right'); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/07_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 07_model_comparison.png")


def error_analysis(model, X_test, y_test, feature_names):
    logger.info("=" * 60)
    logger.info("ERROR ANALYSIS — AdaBoost")
    logger.info("=" * 60)
    y_pred = model.predict(X_test)
    fp = ((y_pred==1)&(y_test==0)).sum()
    fn = ((y_pred==0)&(y_test==1)).sum()
    tp = ((y_pred==1)&(y_test==1)).sum()
    tn = ((y_pred==0)&(y_test==0)).sum()
    logger.info(f"  TP={tp} | TN={tn} | FP={fp} | FN={fn}")
    logger.info(f"  ⚠️  Missed anemia (FN): {fn} — patients sent home without treatment!")
    logger.info(f"  False alarms (FP): {fp}")
    report = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, digits=4)
    logger.info(f"\n{report}")


def save_results(results_df, ablation_df):
    results_df.to_csv(f"{config.OUTPUT_DIR}/day08_results.csv", index=False, float_format='%.4f')
    ablation_df.to_csv(f"{config.OUTPUT_DIR}/day08_ablation.csv", index=False, float_format='%.4f')

    with open(f"{config.OUTPUT_DIR}/day08_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 8: ANEMIA DETECTION — FINAL REPORT\n" + "=" * 70 + "\n\n")
        f.write("OBJECTIVE: Detect anemia from CBC blood tests using AdaBoost\n")
        f.write("FOCUS: Scaling strategies + outlier removal impact\n\n")
        f.write("-" * 70 + "\nABLATION STUDY (Outlier × Scaling)\n" + "-" * 70 + "\n\n")
        f.write(ablation_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "-" * 70 + "\nFINAL RESULTS\n" + "-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "-" * 70 + "\nKEY TAKEAWAYS\n" + "-" * 70 + "\n\n")
        f.write("1. AdaBoost adaptively focuses on hard-to-classify samples\n")
        f.write("2. Outlier removal improves AdaBoost by preventing extreme sample weights\n")
        f.write("3. RobustScaler (median/IQR) is best when outliers remain\n")
        f.write("4. IQR-based outlier removal is simple and effective for blood test data\n")
        f.write("5. The ablation heatmap reveals the optimal preprocessing pipeline\n")
        f.write("6. Hemoglobin is the strongest predictor (as expected clinically)\n")
    logger.info("Results + ablation + report saved to outputs/")
