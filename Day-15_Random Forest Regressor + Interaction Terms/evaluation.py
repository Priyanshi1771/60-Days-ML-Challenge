"""Day 15: BMI Prediction — Evaluation"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config

logger = logging.getLogger(__name__)

BMI_CATEGORIES = [(0, 18.5, "Underweight", "#4FC3F7"),
                   (18.5, 25, "Normal", "#66BB6A"),
                   (25, 30, "Overweight", "#FFB74D"),
                   (30, 35, "Obese I", "#FF7043"),
                   (35, 100, "Obese II+", "#EF5350")]


def evaluate_all(rf_model, baselines, nn_model, X_test, y_test, feature_names):
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    results = []

    # RF
    y_rf = rf_model.predict(X_test).astype(np.float32)
    results.append(_m("Random Forest (Tuned)", y_test, y_rf))

    # Baselines
    for name, data in baselines.items():
        y_p = data["model"].predict(X_test).astype(np.float32)
        results.append(_m(name, y_test, y_p))

    # GPU NN
    nn_model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).to(config.DEVICE)
        with autocast(device_type=config.DEVICE, enabled=config.DEVICE == "cuda"):
            y_nn = nn_model(X_t).squeeze(-1).cpu().numpy()
    results.append(_m("GPU Neural Net", y_test, y_nn))

    # BMI category accuracy (clinical relevance)
    for res, y_pred in zip(results, [y_rf] + [baselines[n]["model"].predict(X_test) for n in baselines] + [y_nn]):
        cat_acc = _category_accuracy(y_test, y_pred)
        res["Category Acc"] = cat_acc

    df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    logger.info(f"\n{'='*60}\nRANKINGS\n{'='*60}")
    for i, row in df.iterrows():
        m = ["🥇", "🥈", "🥉", "  "][min(i, 3)]
        logger.info(f"  {m} {row['Model']:25s} | RMSE={row['RMSE']:.3f} | R²={row['R²']:.4f} | CatAcc={row['Category Acc']:.1f}%")

    # Feature importance from RF
    _plot_rf_importance(rf_model, feature_names)

    return df, y_rf, y_nn


def _m(name, y_true, y_pred):
    return {"Model": name, "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R²": r2_score(y_true, y_pred)}


def _category_accuracy(y_true, y_pred):
    """What % of patients land in the correct BMI category?"""
    def _cat(v):
        for lo, hi, _, _ in BMI_CATEGORIES:
            if lo <= v < hi:
                return _
        return "Obese II+"
    true_cats = [_cat(v) for v in y_true]
    pred_cats = [_cat(v) for v in y_pred]
    return np.mean([t == p for t, p in zip(true_cats, pred_cats)]) * 100


def _plot_rf_importance(model, feature_names):
    fi = model.feature_importances_
    sorted_idx = np.argsort(fi)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    n_raw = len(config.RAW_FEATURES)
    colors = ['#FF7043' if i >= n_raw else '#4FC3F7' for i in sorted_idx]
    ax.barh(range(20), fi[sorted_idx], color=colors, edgecolor='white')
    ax.set_yticks(range(20))
    ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in sorted_idx], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Gini Importance")
    ax.set_title("🌲 RF Feature Importance\n(🟠 Orange = Engineered Interaction Terms)", fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_rf_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_rf_importance.png")


def plot_predictions(y_test, y_rf, y_nn):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, y_pred, name, color in [
        (axes[0, 0], y_rf, f"RF (R²={r2_score(y_test, y_rf):.4f})", '#66BB6A'),
        (axes[0, 1], y_nn, f"NN (R²={r2_score(y_test, y_nn):.4f})", '#4FC3F7')]:
        ax.scatter(y_test, y_pred, alpha=0.2, s=10, color=color, rasterized=True)
        ax.plot([14, 52], [14, 52], 'r--', lw=2)
        # BMI category boundaries
        for thresh in [18.5, 25, 30, 35]:
            ax.axhline(thresh, color='gray', lw=0.5, alpha=0.3)
            ax.axvline(thresh, color='gray', lw=0.5, alpha=0.3)
        ax.set_xlabel("Actual BMI"); ax.set_ylabel("Predicted BMI")
        ax.set_title(f"⚖️ {name}", fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)

    # Residuals
    for ax, y_pred, name, color in [
        (axes[1, 0], y_rf, "RF Residuals", '#FF7043'),
        (axes[1, 1], y_nn, "NN Residuals", '#AB47BC')]:
        resid = y_test - y_pred
        ax.scatter(y_pred, resid, alpha=0.2, s=10, color=color, rasterized=True)
        ax.axhline(0, color='k', lw=1, linestyle='--')
        ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
        ax.set_title(f"📊 {name}", fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_predictions.png")


def plot_bmi_categories(y_test, y_rf):
    """Show prediction accuracy within each BMI category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (0) Error by BMI category
    cat_names, cat_errors = [], []
    for lo, hi, name, color in BMI_CATEGORIES:
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() > 5:
            errors = np.abs(y_test[mask] - y_rf[mask])
            cat_names.append(f"{name}\n(n={mask.sum()})")
            cat_errors.append(errors)

    bp = axes[0].boxplot(cat_errors, labels=cat_names, patch_artist=True, vert=True)
    for patch, (_, _, _, color) in zip(bp['boxes'], BMI_CATEGORIES[:len(cat_errors)]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[0].set_ylabel("Absolute Error (BMI units)")
    axes[0].set_title("📊 Prediction Error by BMI Category\n(Where does the model struggle?)", fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)

    # (1) Confusion matrix style: actual vs predicted category
    def _assign_cat(vals):
        cats = np.zeros(len(vals), dtype=int)
        for i, (lo, hi, _, _) in enumerate(BMI_CATEGORIES):
            cats[(vals >= lo) & (vals < hi)] = i
        return cats

    true_cats = _assign_cat(y_test)
    pred_cats = _assign_cat(y_rf)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_cats, pred_cats, labels=range(5))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    im = axes[1].imshow(cm_pct, cmap='YlOrRd', vmin=0, vmax=100)
    cat_labels = [c[2] for c in BMI_CATEGORIES]
    axes[1].set_xticks(range(5)); axes[1].set_xticklabels(cat_labels, fontsize=8, rotation=30)
    axes[1].set_yticks(range(5)); axes[1].set_yticklabels(cat_labels, fontsize=8)
    axes[1].set_xlabel("Predicted Category"); axes[1].set_ylabel("Actual Category")
    for i in range(5):
        for j in range(5):
            color = 'white' if cm_pct[i, j] > 50 else 'black'
            axes[1].text(j, i, f'{cm_pct[i,j]:.0f}%', ha='center', va='center', fontsize=9, color=color)
    axes[1].set_title("⚖️ BMI Category Accuracy (%)\n(diagonal = correct category)", fontweight='bold')
    plt.colorbar(im, ax=axes[1], label='%')

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/07_bmi_categories.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 07_bmi_categories.png")


def plot_comparison(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#66BB6A', '#4FC3F7', '#FF7043', '#AB47BC'][:len(results_df)]
    ax.barh(range(len(results_df)), results_df["R²"], color=colors, edgecolor='white')
    for i, row in results_df.iterrows():
        ax.text(max(0, row["R²"]) + 0.005, i,
                f'R²={row["R²"]:.4f} | RMSE={row["RMSE"]:.2f} | CatAcc={row["Category Acc"]:.0f}%',
                va='center', fontsize=9)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df["Model"], fontsize=10)
    ax.set_xlabel("R²"); ax.set_title("⚖️ Model Comparison — BMI Prediction", fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/08_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 08_comparison.png")


def save_results(results_df):
    results_df.to_csv(f"{config.OUTPUT_DIR}/day15_results.csv", index=False, float_format='%.4f')
    with open(f"{config.OUTPUT_DIR}/day15_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 15: BMI PREDICTION — FINAL REPORT\n" + "=" * 70 + "\n\n")
        f.write("FOCUS: Interaction feature engineering + Random Forest Regressor\n\n")
        f.write("-" * 70 + "\nRESULTS\n" + "-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "-" * 70 + "\nKEY TAKEAWAYS\n" + "-" * 70 + "\n\n")
        f.write("1. Interaction terms capture combined effects (waist/hip > waist alone)\n")
        f.write("2. Random Forest naturally handles interactions via tree splits\n")
        f.write("3. Engineered interactions can STILL help RF by making patterns explicit\n")
        f.write("4. BMI category accuracy is more clinically useful than raw RMSE\n")
        f.write("5. Errors are larger for obese patients (wider variance in measurements)\n")
        f.write("6. GPU neural net benefits from interaction features too\n")
        f.write("7. Feature importance reveals which interactions the model actually uses\n")
    logger.info("Results saved")
