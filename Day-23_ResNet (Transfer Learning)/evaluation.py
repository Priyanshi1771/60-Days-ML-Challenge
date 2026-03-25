"""Day 23: Skin Lesion Classification — Evaluation"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
    f1_score, ConfusionMatrixDisplay)
import config

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_all_strategies(results, test_loader):
    """Evaluate all 3 strategies on the same test set."""
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)

    # Get ground truth once
    all_labels = []
    for _, lbls in test_loader:
        all_labels.append(lbls.numpy())
    y_test = np.concatenate(all_labels)

    rows = []
    all_preds = {}

    for name, data in results.items():
        model = data["model"]; model.eval()
        preds_list = []
        for imgs, _ in test_loader:
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            out = model(imgs)
            preds_list.append(out.argmax(1).cpu().numpy())

        y_pred = np.concatenate(preds_list)
        all_preds[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average='weighted')
        f1_m = f1_score(y_test, y_pred, average='macro')

        rows.append({"Strategy": name, "Accuracy": acc, "F1 (weighted)": f1_w,
                      "F1 (macro)": f1_m, "Time (s)": data["time"]})
        logger.info(f"  {name:10s} | Acc={acc:.4f} | F1w={f1_w:.4f} | F1m={f1_m:.4f}")

    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    # Best model detailed report
    best_name = df.iloc[0]["Strategy"]
    logger.info(f"\n--- Best Strategy: {best_name.upper()} ---")
    logger.info(f"\n{classification_report(y_test, all_preds[best_name], target_names=config.CLASS_NAMES, digits=4)}")

    _plot_confusion_matrices(y_test, all_preds)
    _plot_per_class_accuracy(y_test, all_preds)
    _plot_final_comparison(df)

    return df, all_preds


def _plot_confusion_matrices(y_test, all_preds):
    n = len(all_preds)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1: axes = [axes]

    for idx, (name, y_pred) in enumerate(all_preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=[c[:8] for c in config.CLASS_NAMES]).plot(
            ax=axes[idx], cmap=plt.cm.Purples, colorbar=False, values_format='d')
        acc = accuracy_score(y_test, y_pred)
        axes[idx].set_title(f"{name} (Acc={acc:.3f})", fontsize=11, fontweight='bold')
        axes[idx].set_xticklabels([c[:8] for c in config.CLASS_NAMES], rotation=45, ha='right', fontsize=7)
        axes[idx].set_yticklabels([c[:8] for c in config.CLASS_NAMES], fontsize=7)

    plt.suptitle("Confusion Matrices -- 3 Transfer Learning Strategies", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_confusion_matrices.png")


def _plot_per_class_accuracy(y_test, all_preds):
    """Show which classes benefit most from transfer learning."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"scratch": "#EF5350", "frozen": "#4FC3F7", "finetune": "#66BB6A"}
    x = np.arange(config.NUM_CLASSES)
    width = 0.25

    for i, (name, y_pred) in enumerate(all_preds.items()):
        per_class_acc = []
        for cls in range(config.NUM_CLASSES):
            mask = y_test == cls
            if mask.sum() > 0:
                per_class_acc.append((y_pred[mask] == cls).mean())
            else:
                per_class_acc.append(0)
        ax.bar(x + i * width, per_class_acc, width, color=colors.get(name, '#999'),
               label=name, edgecolor='white')

    ax.set_xticks(x + width)
    ax.set_xticklabels([c[:12] for c in config.CLASS_NAMES], fontsize=8, rotation=25)
    ax.set_ylabel("Per-Class Accuracy")
    ax.set_title("Per-Class Accuracy by Strategy\n(Transfer learning helps rare classes most!)", fontweight='bold')
    ax.legend(); ax.set_ylim(0, 1.05)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_per_class.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_per_class.png")


def _plot_final_comparison(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#66BB6A', '#4FC3F7', '#EF5350'][:len(df)]

    # Accuracy bars
    axes[0].barh(df["Strategy"], df["Accuracy"], color=colors, edgecolor='white')
    for i, row in df.iterrows():
        axes[0].text(row["Accuracy"] + 0.005, i,
                     f'{row["Accuracy"]:.4f}', va='center', fontweight='bold')
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Accuracy by Strategy", fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)

    # Speed vs accuracy scatter
    for i, row in df.iterrows():
        axes[1].scatter(row["Time (s)"], row["Accuracy"], s=200,
                        color=colors[i], edgecolors='white', linewidth=2, zorder=5)
        axes[1].annotate(row["Strategy"], (row["Time (s)"], row["Accuracy"]),
                         fontsize=10, ha='center', va='bottom', xytext=(0, 10),
                         textcoords='offset points', fontweight='bold')
    axes[1].set_xlabel("Training Time (seconds)")
    axes[1].set_ylabel("Best Val Accuracy")
    axes[1].set_title("Speed vs Accuracy Tradeoff", fontweight='bold')
    axes[1].grid(alpha=0.3); axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_final_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_final_comparison.png")


def save_results(df):
    df.to_csv(f"{config.OUTPUT_DIR}/day23_results.csv", index=False, float_format='%.4f')
    with open(f"{config.OUTPUT_DIR}/day23_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 23: SKIN LESION CLASSIFICATION -- TRANSFER LEARNING\n" + "=" * 70 + "\n\n")
        f.write("FIRST TRANSFER LEARNING PROJECT!\n\n")
        f.write("3 strategies compared: scratch | frozen | finetune\n\n")
        f.write(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\nKEY TAKEAWAYS:\n")
        f.write("1. Pretrained ResNet (finetune) usually beats training from scratch\n")
        f.write("2. Even frozen features (trained on ImageNet) transfer to medical images\n")
        f.write("3. Fine-tuning with low LR preserves pretrained knowledge while adapting\n")
        f.write("4. Transfer learning needs LESS data and FEWER epochs to converge\n")
        f.write("5. ImageNet normalization (mean/std) is required for pretrained models\n")
        f.write("6. Per-class analysis shows transfer helps rare classes most\n")
        f.write("7. This is the foundation for Days 24-28 (VGG, DenseNet, Inception)\n")
    logger.info("Results saved")
