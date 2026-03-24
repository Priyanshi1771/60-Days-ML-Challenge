"""Day 17: Readmission Risk — Evaluation"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve
import config

logger = logging.getLogger(__name__)


def evaluate_all(lr_model, baselines, nn_model, X_test, y_test):
    logger.info("=" * 60)
    logger.info("EVALUATION (Temporal Test Set)")
    logger.info("=" * 60)

    all_models = {"Logistic Regression": lr_model}
    for name, data in baselines.items():
        all_models[name] = data["model"]

    results = []
    proba_dict = {}

    for name, model in all_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        proba_dict[name] = y_proba
        results.append(_m(name, y_test, y_pred, y_proba))

    # GPU NN
    nn_model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).to(config.DEVICE)
        logits = nn_model(X_t).squeeze(-1).cpu().numpy()
        y_nn_proba = 1 / (1 + np.exp(-logits))  # sigmoid
        y_nn_pred = (y_nn_proba >= 0.5).astype(int)
    proba_dict["GPU Neural Net"] = y_nn_proba
    results.append(_m("GPU Neural Net", y_test, y_nn_pred, y_nn_proba))

    df = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)

    logger.info(f"\n{'='*60}\nRANKINGS (by AUC-ROC)\n{'='*60}")
    for i, row in df.iterrows():
        m = ["🥇","🥈","🥉","  "][min(i, 3)]
        logger.info(f"  {m} {row['Model']:22s} | AUC={row['AUC-ROC']:.4f} | F1={row['F1']:.4f} | Recall={row['Recall']:.4f}")

    return df, proba_dict, all_models


def _m(name, y_true, y_pred, y_proba):
    return {"Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred),
            "AUC-ROC": roc_auc_score(y_true, y_proba),
            "Avg Precision": average_precision_score(y_true, y_proba)}


def plot_roc_and_pr(proba_dict, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['#EF5350', '#4FC3F7', '#66BB6A', '#AB47BC']

    for i, (name, proba) in enumerate(proba_dict.items()):
        color = colors[i % len(colors)]
        lw = 3 if "Logistic" in name else 1.8

        # ROC
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        axes[0].plot(fpr, tpr, color=color, lw=lw, label=f"{name} ({auc:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        axes[1].plot(rec, prec, color=color, lw=lw, label=f"{name} ({ap:.3f})")

    axes[0].plot([0,1],[0,1],'k--',alpha=0.3)
    axes[0].set_title("📈 ROC Curves", fontweight='bold', fontsize=13)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.2)
    axes[0].spines[['top','right']].set_visible(False)

    baseline = y_test.mean()
    axes[1].axhline(baseline, color='k', linestyle='--', alpha=0.3, label=f'Baseline ({baseline:.3f})')
    axes[1].set_title("🎯 Precision-Recall Curves", fontweight='bold', fontsize=13)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.2)
    axes[1].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_roc_pr.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_roc_pr.png")


def plot_calibration(proba_dict, y_test):
    """Calibration curve: are predicted probabilities accurate?"""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ['#EF5350', '#4FC3F7', '#66BB6A', '#AB47BC']

    for i, (name, proba) in enumerate(proba_dict.items()):
        fraction_pos, mean_predicted = calibration_curve(y_test, proba, n_bins=10, strategy='uniform')
        ax.plot(mean_predicted, fraction_pos, 'o-', color=colors[i % len(colors)], lw=2, label=name, markersize=6)

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Perfectly calibrated')
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("📊 Calibration Curve — Are Predicted Probabilities Trustworthy?\n"
                 "(Closer to diagonal = better calibrated)", fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_calibration.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_calibration.png")


def plot_confusion_matrices(all_models, nn_model, X_test, y_test, proba_dict):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    all_with_nn = dict(all_models)

    # NN predictions
    nn_pred = (proba_dict["GPU Neural Net"] >= 0.5).astype(int)

    for idx, (name, ax) in enumerate(zip(list(all_with_nn.keys()) + ["GPU Neural Net"], axes)):
        if name == "GPU Neural Net":
            y_pred = nn_pred
        else:
            y_pred = all_with_nn[name].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Stay", "Readmit"]).plot(
            ax=ax, cmap=plt.cm.OrRd, colorbar=False, values_format='d')
        ax.set_title(name[:18], fontsize=9, fontweight='bold')

    fig.suptitle("🏥 Confusion Matrices — Readmission Prediction", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/07_confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 07_confusion_matrices.png")


def plot_comparison(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#EF5350', '#4FC3F7', '#66BB6A', '#AB47BC'][:len(results_df)]
    ax.barh(range(len(results_df)), results_df["AUC-ROC"], color=colors, edgecolor='white')
    for i, row in results_df.iterrows():
        ax.text(row["AUC-ROC"] + 0.005, i,
                f'AUC={row["AUC-ROC"]:.4f} | F1={row["F1"]:.4f}', va='center', fontsize=9)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df["Model"], fontsize=10)
    ax.set_xlabel("AUC-ROC")
    ax.set_title("🏥 Model Comparison — Readmission Risk", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/08_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 08_comparison.png")


def save_results(results_df, split_comparison):
    results_df.to_csv(f"{config.OUTPUT_DIR}/day17_results.csv", index=False, float_format='%.4f')
    with open(f"{config.OUTPUT_DIR}/day17_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 17: HOSPITAL READMISSION RISK — FINAL REPORT\n" + "=" * 70 + "\n\n")
        f.write("FOCUS: Time-based splits + deployment reality for healthcare ML\n\n")
        f.write("-" * 70 + "\nTEMPORAL vs RANDOM SPLIT COMPARISON\n" + "-" * 70 + "\n\n")
        for name, data in split_comparison.items():
            f.write(f"  {name:25s} | AUC={data['auc']:.4f} | F1={data['f1']:.4f}\n")
        f.write("\n" + "-" * 70 + "\nFINAL RESULTS (Temporal Split)\n" + "-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "-" * 70 + "\nKEY TAKEAWAYS\n" + "-" * 70 + "\n\n")
        f.write("1. TEMPORAL SPLIT is mandatory for healthcare deployment models\n")
        f.write("2. Random split LEAKS future data → inflated AUC (misleading!)\n")
        f.write("3. Temporal drift: readmission rates change over time (policy changes)\n")
        f.write("4. Calibration matters: predicted 20% risk should mean ~20% actual readmission\n")
        f.write("5. Previous admissions + comorbidity are the strongest risk factors\n")
        f.write("6. Class imbalance (18% readmit) requires balanced weighting or SMOTE\n")
        f.write("7. LR coefficients are directly interpretable for clinical staff\n")
        f.write("8. In healthcare: Recall > Precision (missing a readmission = worse than false alarm)\n")
    logger.info("Results saved")
