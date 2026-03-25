"""Day 21: Pneumonia Detection — Evaluation"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
    f1_score, accuracy_score, precision_score, recall_score,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve, average_precision_score)
import config

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(model, test_loader):
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)

    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for imgs, labels in test_loader:
        imgs = imgs.to(config.DEVICE, non_blocking=True)
        out = model(imgs)
        probs = F.softmax(out, dim=1)
        all_preds.append(out.argmax(1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_probs)
    y_test = np.concatenate(all_labels)
    y_proba_pos = y_proba[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba_pos)
    ap = average_precision_score(y_test, y_proba_pos)

    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  F1:        {f1:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  AUC-ROC:   {auc:.4f}")
    logger.info(f"  Avg Prec:  {ap:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, digits=4)}")

    # ─── Plot 1: Confusion Matrix + ROC + PR Curve ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES).plot(
        ax=axes[0], cmap=plt.cm.Blues, colorbar=False, values_format='d')
    axes[0].set_title("Confusion Matrix", fontweight='bold')

    fpr, tpr, _ = roc_curve(y_test, y_proba_pos)
    axes[1].plot(fpr, tpr, color='#4FC3F7', lw=3, label=f'CNN (AUC={auc:.3f})')
    axes[1].fill_between(fpr, tpr, alpha=0.1, color='#4FC3F7')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve", fontweight='bold')
    axes[1].legend(fontsize=11); axes[1].grid(alpha=0.2)
    axes[1].spines[['top', 'right']].set_visible(False)

    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_proba_pos)
    axes[2].plot(pr_rec, pr_prec, color='#66BB6A', lw=3, label=f'CNN (AP={ap:.3f})')
    axes[2].fill_between(pr_rec, pr_prec, alpha=0.1, color='#66BB6A')
    axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
    axes[2].set_title("Precision-Recall Curve", fontweight='bold')
    axes[2].legend(fontsize=11); axes[2].grid(alpha=0.2)
    axes[2].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_evaluation.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_evaluation.png")

    # ─── Plot 2: Confidence + Error Analysis ────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    max_prob = y_proba.max(axis=1)
    correct = y_pred == y_test

    axes[0].hist(max_prob[correct], bins=20, alpha=0.7, color='#66BB6A', label='Correct', edgecolor='white')
    axes[0].hist(max_prob[~correct], bins=20, alpha=0.7, color='#EF5350', label='Wrong', edgecolor='white')
    axes[0].set_title("Prediction Confidence", fontweight='bold')
    axes[0].set_xlabel("Max Probability"); axes[0].legend()
    axes[0].spines[['top', 'right']].set_visible(False)

    # Per-class confidence
    for cls, name, color in [(0, "Normal", "#4FC3F7"), (1, "Pneumonia", "#EF5350")]:
        mask = y_test == cls
        axes[1].hist(y_proba_pos[mask], bins=20, alpha=0.6, color=color, label=name, edgecolor='white')
    axes[1].set_title("P(Pneumonia) by True Class", fontweight='bold')
    axes[1].set_xlabel("P(Pneumonia)"); axes[1].legend()
    axes[1].spines[['top', 'right']].set_visible(False)

    # FP vs FN analysis
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    tn = ((y_pred == 0) & (y_test == 0)).sum()
    axes[2].bar(["TP\n(correct\npneumonia)", "TN\n(correct\nnormal)",
                  "FP\n(false\nalarm)", "FN\n(missed\npneumonia)"],
                [tp, tn, fp, fn], color=['#66BB6A', '#4FC3F7', '#FFB74D', '#EF5350'], edgecolor='white')
    for i, v in enumerate([tp, tn, fp, fn]):
        axes[2].text(i, v + 1, str(v), ha='center', fontweight='bold')
    axes[2].set_title("Error Breakdown", fontweight='bold')
    axes[2].spines[['top', 'right']].set_visible(False)

    logger.info(f"  TP={tp} | TN={tn} | FP={fp} | FN={fn}")
    logger.info(f"  Missed pneumonia (FN): {fn} -- dangerous!")
    logger.info(f"  False alarms (FP): {fp}")

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_error_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_error_analysis.png")

    # ─── Save Results ───────────────────────────────────────────
    results = {"Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec,
               "AUC-ROC": auc, "Avg_Precision": ap}

    pd.DataFrame([results]).to_csv(
        f"{config.OUTPUT_DIR}/day21_results.csv", index=False, float_format='%.4f')

    with open(f"{config.OUTPUT_DIR}/day21_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 21: PNEUMONIA DETECTION -- FINAL REPORT\n" + "=" * 70 + "\n\n")
        f.write("PHASE 3 BEGINS: Deep Learning & Medical Imaging!\n\n")
        f.write(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\n")
        f.write(f"Precision: {prec:.4f} | Recall: {rec:.4f} | AP: {ap:.4f}\n\n")
        f.write(f"TP={tp} | TN={tn} | FP={fp} | FN={fn}\n\n")
        f.write("KEY TAKEAWAYS:\n")
        f.write("1. 4-block CNN captures hierarchical features in chest X-rays\n")
        f.write("2. Data augmentation (flip, rotate, translate, jitter) prevents overfitting\n")
        f.write("3. Grayscale input (1 channel) is sufficient for X-ray classification\n")
        f.write("4. In pneumonia detection: FN (missed) is worse than FP (false alarm)\n")
        f.write("5. Recall is the priority metric for screening applications\n")
        f.write("6. AdaptiveAvgPool makes the CNN flexible to different input sizes\n")
        f.write("7. Dropout(0.5) is aggressive but needed for small medical datasets\n")
        f.write("8. This is the foundation for Days 22-30 (segmentation, transfer learning)\n")
    logger.info("Results saved")

    return results
