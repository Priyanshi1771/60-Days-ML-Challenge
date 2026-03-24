"""
=============================================================================
 Day 5: Thyroid Disease Classification — Evaluation
=============================================================================
 Comprehensive evaluation suite:
   1. Per-model classification reports
   2. Confusion matrices (individual + ensemble)
   3. Model comparison bar chart (CV F1 scores)
   4. Ensemble vs Individual performance analysis
   5. Error analysis on misclassified samples
   6. Voting agreement analysis
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score,
    ConfusionMatrixDisplay
)

import config

logger = logging.getLogger(__name__)


def evaluate_all_models(gnb, classifiers, ensembles, X_test, y_test,
                        gnb_cv, individual_cv, ensemble_cv):
    """
    Run full evaluation on all models and produce comparison.
    
    Returns a summary DataFrame with all metrics.
    """
    logger.info("=" * 60)
    logger.info("EVALUATION ON HELD-OUT TEST SET")
    logger.info("=" * 60)
    
    # Collect all models
    all_models = {
        "Gaussian NB": gnb,
        "Logistic Regression": classifiers["lr"],
        "SVM (RBF)": classifiers["svm"],
        "Random Forest": classifiers["rf"],
        "KNN": classifiers["knn"],
        "Hard Voting": ensembles["hard_voting"],
        "Soft Voting": ensembles["soft_voting"],
        "Weighted Voting": ensembles["weighted_voting"]
    }
    
    # Collect CV scores
    all_cv = {
        "Gaussian NB": gnb_cv,
        "Logistic Regression": individual_cv["lr"],
        "SVM (RBF)": individual_cv["svm"],
        "Random Forest": individual_cv["rf"],
        "KNN": individual_cv["knn"],
        "Hard Voting": ensemble_cv["hard_voting"],
        "Soft Voting": ensemble_cv["soft_voting"],
        "Weighted Voting": ensemble_cv["weighted_voting"]
    }
    
    results = []
    
    for name, model in all_models.items():
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average='weighted')
        f1_m = f1_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        cv_mean = all_cv[name].mean()
        cv_std = all_cv[name].std()
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1 (Weighted)": f1_w,
            "F1 (Macro)": f1_m,
            "Precision": prec,
            "Recall": rec,
            "CV F1 Mean": cv_mean,
            "CV F1 Std": cv_std
        })
        
        logger.info(f"\n{'─'*50}")
        logger.info(f"  {name}")
        logger.info(f"{'─'*50}")
        logger.info(f"  Accuracy:       {acc:.4f}")
        logger.info(f"  F1 (weighted):  {f1_w:.4f}")
        logger.info(f"  F1 (macro):     {f1_m:.4f}")
        logger.info(f"  Precision:      {prec:.4f}")
        logger.info(f"  Recall:         {rec:.4f}")
        logger.info(f"  CV F1:          {cv_mean:.4f} ± {cv_std:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("F1 (Weighted)", ascending=False).reset_index(drop=True)
    
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RANKINGS (by Test F1 Weighted)")
    logger.info(f"{'='*60}")
    for i, row in results_df.iterrows():
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        logger.info(f"  {emoji} {row['Model']:25s} | F1={row['F1 (Weighted)']:.4f} | Acc={row['Accuracy']:.4f}")
    
    return results_df, all_models


def plot_confusion_matrices(all_models, X_test, y_test):
    """Plot confusion matrices for all 8 models in a grid."""
    logger.info("Generating confusion matrices...")
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.ravel()
    
    cmap = plt.cm.Blues
    
    for idx, (name, model) in enumerate(all_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        disp = ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES)
        disp.plot(ax=axes[idx], cmap=cmap, colorbar=False, values_format='d')
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
    
    fig.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_confusion_matrices.png")


def plot_model_comparison(results_df):
    """Create a comprehensive model comparison visualization."""
    logger.info("Generating model comparison chart...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # ─── Left: Test Set Metrics ──────────────────────────────────────────
    models = results_df["Model"]
    x = np.arange(len(models))
    width = 0.25
    
    colors = {
        "Accuracy": "#4FC3F7",
        "F1 (Weighted)": "#FF7043",
        "F1 (Macro)": "#66BB6A"
    }
    
    for i, (metric, color) in enumerate(colors.items()):
        values = results_df[metric].values
        bars = axes[0].barh(x + i * width, values, width, label=metric, color=color, alpha=0.85)
    
    axes[0].set_yticks(x + width)
    axes[0].set_yticklabels(models, fontsize=10)
    axes[0].set_xlabel("Score", fontsize=12)
    axes[0].set_title("Test Set Performance", fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].set_xlim(0, 1.05)
    axes[0].axvline(x=0.9, color='gray', linestyle='--', alpha=0.3)
    axes[0].axvline(x=0.95, color='gray', linestyle='--', alpha=0.3)
    axes[0].spines[['top', 'right']].set_visible(False)
    
    # ─── Right: CV F1 with Error Bars ────────────────────────────────────
    # Color ensemble models differently
    bar_colors = []
    for name in results_df["Model"]:
        if "Voting" in name:
            bar_colors.append("#FF7043")
        elif name == "Gaussian NB":
            bar_colors.append("#AB47BC")
        else:
            bar_colors.append("#4FC3F7")
    
    axes[1].barh(
        results_df["Model"],
        results_df["CV F1 Mean"],
        xerr=results_df["CV F1 Std"],
        color=bar_colors,
        alpha=0.85,
        capsize=5,
        edgecolor='white',
        linewidth=0.8
    )
    axes[1].set_xlabel("CV F1 (Weighted) ± Std", fontsize=12)
    axes[1].set_title("Cross-Validation Performance", fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1.05)
    axes[1].axvline(x=0.9, color='gray', linestyle='--', alpha=0.3)
    axes[1].spines[['top', 'right']].set_visible(False)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#AB47BC', alpha=0.85, label='Baseline (NB)'),
        Patch(facecolor='#4FC3F7', alpha=0.85, label='Individual Models'),
        Patch(facecolor='#FF7043', alpha=0.85, label='Ensemble Models')
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_model_comparison.png")


def plot_ensemble_advantage(results_df):
    """
    Visualize how ensemble models compare to their individual components.
    Shows the 'wisdom of crowds' effect.
    """
    logger.info("Generating ensemble advantage chart...")
    
    individual_names = ["Logistic Regression", "SVM (RBF)", "Random Forest", "KNN"]
    ensemble_names = ["Hard Voting", "Soft Voting", "Weighted Voting"]
    
    individual_f1 = results_df[results_df["Model"].isin(individual_names)]["F1 (Weighted)"].values
    ensemble_f1 = results_df[results_df["Model"].isin(ensemble_names)]["F1 (Weighted)"].values
    nb_f1 = results_df[results_df["Model"] == "Gaussian NB"]["F1 (Weighted)"].values[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual models
    ax.scatter(range(len(individual_names)), individual_f1, 
               s=200, c='#4FC3F7', marker='o', zorder=5, label='Individual Models', edgecolors='white', linewidth=2)
    for i, (name, score) in enumerate(zip(individual_names, individual_f1)):
        ax.annotate(name, (i, score), textcoords="offset points", xytext=(0, 15),
                    ha='center', fontsize=9, fontweight='bold')
    
    # Plot ensemble models
    offset = len(individual_names) + 0.5
    ax.scatter([offset + i for i in range(len(ensemble_names))], ensemble_f1,
               s=250, c='#FF7043', marker='*', zorder=5, label='Ensemble Models', edgecolors='white', linewidth=2)
    for i, (name, score) in enumerate(zip(ensemble_names, ensemble_f1)):
        ax.annotate(name, (offset + i, score), textcoords="offset points", xytext=(0, 15),
                    ha='center', fontsize=9, fontweight='bold')
    
    # Plot NB baseline
    ax.axhline(y=nb_f1, color='#AB47BC', linestyle='--', alpha=0.7, label=f'NB Baseline (F1={nb_f1:.4f})')
    
    # Best individual line
    best_individual = individual_f1.max()
    ax.axhline(y=best_individual, color='#4FC3F7', linestyle=':', alpha=0.5, 
               label=f'Best Individual (F1={best_individual:.4f})')
    
    ax.set_ylabel("F1 Score (Weighted)", fontsize=12)
    ax.set_title("Ensemble Advantage: Individual vs Ensemble Performance", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xticks([])
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_ensemble_advantage.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_ensemble_advantage.png")


def voting_agreement_analysis(classifiers, ensembles, X_test, y_test):
    """
    Analyze where ensemble members agree/disagree.
    
    This reveals:
    - How often all classifiers agree (high-confidence predictions)
    - Which samples cause disagreement (potential hard cases)
    - Whether disagreement correlates with misclassification
    """
    logger.info("-" * 60)
    logger.info("VOTING AGREEMENT ANALYSIS")
    logger.info("-" * 60)
    
    # Get individual predictions
    preds = {}
    for name, clf in classifiers.items():
        preds[name] = clf.predict(X_test)
    
    preds_array = np.array(list(preds.values()))  # shape: (4, n_test)
    
    # Count agreement per sample
    n_test = X_test.shape[0]
    agreement_counts = []
    for i in range(n_test):
        unique_preds = np.unique(preds_array[:, i])
        max_agreement = max(np.sum(preds_array[:, i] == u) for u in unique_preds)
        agreement_counts.append(max_agreement)
    
    agreement_counts = np.array(agreement_counts)
    
    # Analyze
    unanimous = (agreement_counts == 4).sum()
    three_agree = (agreement_counts == 3).sum()
    split = (agreement_counts == 2).sum()
    
    logger.info(f"  Unanimous agreement (4/4): {unanimous}/{n_test} ({unanimous/n_test*100:.1f}%)")
    logger.info(f"  3/4 agreement:             {three_agree}/{n_test} ({three_agree/n_test*100:.1f}%)")
    logger.info(f"  Split vote (2/2):          {split}/{n_test} ({split/n_test*100:.1f}%)")
    
    # Check if disagreement → errors
    ensemble_pred = ensembles["soft_voting"].predict(X_test)
    correct = (ensemble_pred == y_test)
    
    for threshold in [4, 3, 2]:
        mask = agreement_counts == threshold
        if mask.sum() > 0:
            acc = correct[mask].mean()
            logger.info(f"  Accuracy when {threshold}/4 agree: {acc:.4f} ({mask.sum()} samples)")
    
    # Plot agreement distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Unanimous\n(4/4)", "Majority\n(3/4)", "Split\n(2/2)"]
    counts = [unanimous, three_agree, split]
    colors = ["#66BB6A", "#FFB74D", "#EF5350"]
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{count} ({count/n_test*100:.0f}%)', ha='center', fontweight='bold')
    
    ax.set_title("Voting Agreement Among Ensemble Members", fontsize=14, fontweight='bold')
    ax.set_ylabel("Number of Test Samples", fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/07_voting_agreement.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 07_voting_agreement.png")


def error_analysis(best_model, best_name, X_test, y_test, feature_names):
    """
    Detailed error analysis on the best model's misclassifications.
    
    For each misclassified sample:
    - What was the true class vs predicted class?
    - What were the feature values?
    - Are there patterns in the errors?
    """
    logger.info("=" * 60)
    logger.info(f"ERROR ANALYSIS — {best_name}")
    logger.info("=" * 60)
    
    y_pred = best_model.predict(X_test)
    errors = y_pred != y_test
    n_errors = errors.sum()
    
    logger.info(f"  Total misclassifications: {n_errors}/{len(y_test)} ({n_errors/len(y_test)*100:.1f}%)")
    
    if n_errors == 0:
        logger.info("  Perfect classification on test set!")
        return
    
    # Error breakdown by class
    logger.info(f"\n  Error Breakdown:")
    for true_cls in range(3):
        for pred_cls in range(3):
            if true_cls != pred_cls:
                count = ((y_test == true_cls) & (y_pred == pred_cls)).sum()
                if count > 0:
                    logger.info(f"    {config.CLASS_NAMES[true_cls]} → {config.CLASS_NAMES[pred_cls]}: {count} samples")
    
    # Feature analysis of misclassified samples
    error_features = X_test[errors]
    correct_features = X_test[~errors]
    
    logger.info(f"\n  Feature comparison (mean values):")
    logger.info(f"  {'Feature':35s} | {'Correct':>10s} | {'Misclassified':>15s} | {'Diff':>8s}")
    logger.info(f"  {'-'*75}")
    for i, feat in enumerate(feature_names):
        correct_mean = correct_features[:, i].mean()
        error_mean = error_features[:, i].mean()
        diff = error_mean - correct_mean
        logger.info(f"  {feat:35s} | {correct_mean:10.4f} | {error_mean:15.4f} | {diff:+8.4f}")
    
    # Classification report for best model
    logger.info(f"\n  Full Classification Report ({best_name}):")
    report = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, digits=4)
    for line in report.split('\n'):
        logger.info(f"    {line}")


def save_results_report(results_df):
    """Save final results to CSV and formatted text report."""
    
    # CSV
    csv_path = f"{config.OUTPUT_DIR}/day05_results.csv"
    results_df.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"Results saved to: {csv_path}")
    
    # Text report
    report_path = f"{config.OUTPUT_DIR}/day05_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  DAY 5: THYROID DISEASE CLASSIFICATION — FINAL REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OBJECTIVE: Compare Naive Bayes baseline against ensemble voting\n")
        f.write("           classifiers (hard, soft, weighted) for multi-class\n")
        f.write("           thyroid disease diagnosis.\n\n")
        
        f.write("DATASET: UCI New Thyroid (215 samples, 5 features, 3 classes)\n")
        f.write("         Classes: Normal, Hyperthyroid, Hypothyroid\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("RESULTS (sorted by Test F1 Weighted)\n")
        f.write("-" * 70 + "\n\n")
        
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n")
        
        best = results_df.iloc[0]
        f.write(f"BEST MODEL: {best['Model']}\n")
        f.write(f"  Test F1 (Weighted): {best['F1 (Weighted)']:.4f}\n")
        f.write(f"  Test Accuracy:      {best['Accuracy']:.4f}\n")
        f.write(f"  CV F1:              {best['CV F1 Mean']:.4f} ± {best['CV F1 Std']:.4f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("KEY TAKEAWAYS\n")
        f.write("-" * 70 + "\n\n")
        f.write("1. Ensemble voting classifiers combine diverse models to reduce\n")
        f.write("   individual model weaknesses.\n\n")
        f.write("2. Soft voting (probability averaging) typically outperforms hard\n")
        f.write("   voting (majority label) because it uses confidence information.\n\n")
        f.write("3. Weighted voting can further improve by giving more influence to\n")
        f.write("   better-performing base classifiers.\n\n")
        f.write("4. Naive Bayes provides a strong baseline for this dataset due to\n")
        f.write("   the relative independence of thyroid lab measurements.\n\n")
        f.write("5. Model diversity is crucial — ensemble of similar models won't\n")
        f.write("   improve much over a single model.\n")
    
    logger.info(f"Report saved to: {report_path}")
