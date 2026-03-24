"""
=============================================================================
 Day 10: Malaria Cell Classification — Evaluation (Optimized)
=============================================================================
 CNN-SPECIFIC EVALUATION:
   1. Test set metrics (Acc, F1, AUC, Precision, Recall)
   2. Confusion matrix
   3. ROC curve
   4. Feature map visualization (what the CNN "sees")
   5. Misclassified samples analysis
   6. Per-class confidence distribution
=============================================================================
"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, roc_curve)
import config

logger = logging.getLogger(__name__)


@torch.no_grad()
def get_predictions(model, loader):
    """Efficient batch prediction — no grad computation."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    for images, labels in loader:
        images = images.to(config.DEVICE, non_blocking=True)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    
    return (np.concatenate(all_preds),
            np.concatenate(all_probs),
            np.concatenate(all_labels))


def evaluate_model(model, test_loader):
    """Full evaluation on held-out test set."""
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)
    
    y_pred, y_proba, y_test = get_predictions(model, test_loader)
    y_proba_pos = y_proba[:, 1]  # Probability of "Uninfected"
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_proba_pos)
    
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  F1 (wt):   {f1:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  AUC-ROC:   {auc:.4f}")
    
    report = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, digits=4)
    logger.info(f"\n{report}")
    
    results = {"Model": "Custom CNN", "Accuracy": acc, "F1 (Weighted)": f1,
               "Precision": prec, "Recall": rec, "AUC-ROC": auc}
    
    return results, y_pred, y_proba, y_test


def plot_confusion_matrix(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES).plot(
        ax=ax, cmap=plt.cm.Greens, colorbar=False, values_format='d')
    ax.set_title("🦟 Confusion Matrix — CNN Malaria Detection", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_confusion_matrix.png")


def plot_roc_curve(y_test, y_proba):
    fig, ax = plt.subplots(figsize=(8, 7))
    y_proba_pos = y_proba[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba_pos)
    auc_val = roc_auc_score(y_test, y_proba_pos)
    
    ax.plot(fpr, tpr, color='#66BB6A', lw=3, label=f'CNN (AUC = {auc_val:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#66BB6A')
    ax.plot([0,1], [0,1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("🔬 ROC Curve — CNN Malaria Classification", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(alpha=0.2)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_roc_curve.png")


def plot_confidence_distribution(y_test, y_proba):
    """Show model confidence for correct vs incorrect predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    max_probs = y_proba.max(axis=1)
    y_pred = y_proba.argmax(axis=1)
    correct = y_pred == y_test
    
    # Correct predictions
    axes[0].hist(max_probs[correct], bins=20, color='#66BB6A', alpha=0.8, edgecolor='white', label='Correct')
    axes[0].hist(max_probs[~correct], bins=20, color='#EF5350', alpha=0.8, edgecolor='white', label='Wrong')
    axes[0].set_title("🎯 Prediction Confidence Distribution", fontweight='bold')
    axes[0].set_xlabel("Max Probability"); axes[0].set_ylabel("Count")
    axes[0].legend(); axes[0].spines[['top','right']].set_visible(False)
    
    # Per-class
    for cls, name, color in [(0, "Parasitized", "#EF5350"), (1, "Uninfected", "#66BB6A")]:
        mask = y_test == cls
        axes[1].hist(y_proba[mask, cls], bins=20, alpha=0.6, color=color, label=name, edgecolor='white')
    axes[1].set_title("🦟 Per-Class Probability Distribution", fontweight='bold')
    axes[1].set_xlabel("P(true class)"); axes[1].set_ylabel("Count")
    axes[1].legend(); axes[1].spines[['top','right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_confidence_dist.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_confidence_dist.png")


@torch.no_grad()
def visualize_feature_maps(model, test_loader):
    """
    Visualize what the CNN 'sees' at each convolutional layer.
    Shows first 8 feature maps from each of the 3 conv blocks.
    """
    model.eval()
    images, _ = next(iter(test_loader))
    img = images[0:1].to(config.DEVICE)
    
    # Extract intermediate activations
    activations = []
    x = img
    for layer in model.features:
        x = layer(x)
        if isinstance(layer, torch.nn.ReLU):
            activations.append(x.cpu().squeeze(0).numpy())
    
    if len(activations) < 3:
        logger.info("  Skipping feature map visualization (not enough activations)")
        return
    
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    block_names = ["Conv Block 1 (edges/colors)", "Conv Block 2 (textures)", "Conv Block 3 (patterns)"]
    
    for row, (act, name) in enumerate(zip(activations[:3], block_names)):
        n_show = min(8, act.shape[0])
        for col in range(n_show):
            axes[row, col].imshow(act[col], cmap='viridis')
            axes[row, col].axis('off')
        axes[row, 0].set_ylabel(name, fontsize=9, fontweight='bold')
        for col in range(n_show, 8):
            axes[row, col].axis('off')
    
    fig.suptitle("🧠 CNN Feature Maps — What Each Layer Detects\n"
                 "(Early layers → edges | Middle → textures | Deep → cell patterns)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_feature_maps.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_feature_maps.png")


def error_analysis(y_test, y_pred, y_proba):
    """Analyze misclassifications."""
    logger.info("=" * 60)
    logger.info("ERROR ANALYSIS")
    logger.info("=" * 60)
    
    fp = ((y_pred==1)&(y_test==0)).sum()
    fn = ((y_pred==0)&(y_test==1)).sum()
    tp = ((y_pred==1)&(y_test==1)).sum()
    tn = ((y_pred==0)&(y_test==0)).sum()
    
    logger.info(f"  TP={tp} | TN={tn} | FP={fp} | FN={fn}")
    logger.info(f"  ⚠️  FN (missed parasites): {fn} — infected patient sent home!")
    logger.info(f"  FP (false alarm): {fp} — unnecessary treatment")
    
    # Confidence on errors
    wrong = y_pred != y_test
    if wrong.sum() > 0:
        wrong_conf = y_proba.max(axis=1)[wrong]
        logger.info(f"  Error confidence: mean={wrong_conf.mean():.3f}, max={wrong_conf.max():.3f}")
        logger.info(f"  → High-confidence errors are most dangerous!")


def save_results(results):
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{config.OUTPUT_DIR}/day10_results.csv", index=False, float_format='%.4f')
    
    with open(f"{config.OUTPUT_DIR}/day10_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 10: MALARIA CELL CLASSIFICATION — FINAL REPORT\n" + "=" * 70 + "\n\n")
        f.write("🎉 MILESTONE: First deep learning project in the 60-day challenge!\n\n")
        f.write("OBJECTIVE: Classify malaria-infected vs healthy blood cells using CNN\n")
        f.write(f"DATASET: {config.SYNTHETIC_N} cell images ({config.IMG_SIZE}×{config.IMG_SIZE})\n")
        f.write(f"MODEL: Custom 3-block CNN ({count_params()} parameters)\n\n")
        f.write("-" * 70 + "\nRESULTS\n" + "-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "-" * 70 + "\nKEY TAKEAWAYS\n" + "-" * 70 + "\n\n")
        f.write("1. CNNs automatically learn hierarchical features (edges→textures→patterns)\n")
        f.write("2. Data augmentation is essential — prevents overfitting on limited images\n")
        f.write("3. BatchNorm + Dropout together = strong regularization combo\n")
        f.write("4. AdaptiveAvgPool makes architecture flexible to different input sizes\n")
        f.write("5. Early stopping prevents the model from memorizing training data\n")
        f.write("6. Feature maps reveal WHAT the CNN learned — not just accuracy numbers\n")
        f.write("7. This is the FOUNDATION for Days 21-30 (medical imaging deep learning)\n")
    logger.info("Results + report saved")


def count_params():
    """Quick param count string."""
    from model_training import MalariaCNN, count_parameters
    m = MalariaCNN()
    return f"{count_parameters(m):,}"
