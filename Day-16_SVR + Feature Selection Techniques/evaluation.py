"""Day 16: Telomere Length — Evaluation"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config

logger = logging.getLogger(__name__)


def evaluate_all(svr_model, baselines, nn_model, X_test, y_test, best_mask):
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    X_sel = X_test[:, best_mask]
    results = []

    y_svr = svr_model.predict(X_sel).astype(np.float32)
    results.append(_m("SVR (Tuned)", y_test, y_svr))

    preds_dict = {"SVR (Tuned)": y_svr}
    for name, data in baselines.items():
        y_p = data["model"].predict(X_sel).astype(np.float32)
        results.append(_m(name, y_test, y_p))
        preds_dict[name] = y_p

    nn_model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_sel).to(config.DEVICE)
        with autocast(device_type=config.DEVICE, enabled=config.DEVICE == "cuda"):
            y_nn = nn_model(X_t).squeeze(-1).cpu().numpy()
    results.append(_m("GPU Neural Net", y_test, y_nn))
    preds_dict["GPU Neural Net"] = y_nn

    df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    logger.info(f"\n{'='*60}\nRANKINGS\n{'='*60}")
    for i, row in df.iterrows():
        m = ["🥇", "🥈", "🥉", "  "][min(i, 3)]
        logger.info(f"  {m} {row['Model']:20s} | RMSE={row['RMSE']:.4f} | R²={row['R²']:.4f}")

    return df, preds_dict


def _m(name, y_true, y_pred):
    return {"Model": name, "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R²": r2_score(y_true, y_pred)}


def plot_predictions(y_test, preds_dict, X_test):
    best_name = min(preds_dict, key=lambda k: np.sqrt(mean_squared_error(y_test, preds_dict[k])))
    y_best = preds_dict[best_name]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Actual vs predicted — best model
    axes[0, 0].scatter(y_test, y_best, alpha=0.3, s=12, color='#7E57C2', rasterized=True)
    axes[0, 0].plot([3, 14], [3, 14], 'r--', lw=2)
    axes[0, 0].set_xlabel("Actual (kb)"); axes[0, 0].set_ylabel("Predicted (kb)")
    axes[0, 0].set_title(f"🧬 {best_name} (R²={r2_score(y_test, y_best):.4f})", fontweight='bold')
    axes[0, 0].spines[['top', 'right']].set_visible(False)

    # Residuals
    resid = y_test - y_best
    axes[0, 1].scatter(y_best, resid, alpha=0.3, s=12, color='#FF7043', rasterized=True)
    axes[0, 1].axhline(0, color='k', lw=1, linestyle='--')
    axes[0, 1].set_xlabel("Predicted (kb)"); axes[0, 1].set_ylabel("Residual")
    axes[0, 1].set_title("📊 Residuals", fontweight='bold')
    axes[0, 1].spines[['top', 'right']].set_visible(False)

    # Telomere vs Age (colored by prediction error)
    age_col = X_test[:, 0]  # age is first feature
    error = np.abs(y_test - y_best)
    sc = axes[1, 0].scatter(age_col, y_test, c=error, cmap='RdYlGn_r', alpha=0.5, s=15, rasterized=True)
    axes[1, 0].set_xlabel("Age (years)"); axes[1, 0].set_ylabel("Telomere Length (kb)")
    axes[1, 0].set_title("🧬 Telomere vs Age (color = prediction error)", fontweight='bold')
    plt.colorbar(sc, ax=axes[1, 0], label='|Error| (kb)')
    axes[1, 0].spines[['top', 'right']].set_visible(False)

    # Error distribution
    axes[1, 1].hist(resid, bins=40, color='#7E57C2', edgecolor='white', alpha=0.85)
    axes[1, 1].axvline(0, color='k', lw=1.5, linestyle='--')
    axes[1, 1].set_xlabel("Prediction Error (kb)"); axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title(f"📊 Error Distribution (MAE={np.mean(np.abs(resid)):.3f} kb)", fontweight='bold')
    axes[1, 1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_predictions.png")


def plot_comparison(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#7E57C2', '#4FC3F7', '#66BB6A', '#FF7043'][:len(results_df)]
    ax.barh(range(len(results_df)), results_df["R²"], color=colors, edgecolor='white')
    for i, row in results_df.iterrows():
        ax.text(max(0, row["R²"]) + 0.005, i,
                f'R²={row["R²"]:.4f} | RMSE={row["RMSE"]:.4f} kb', va='center', fontsize=9)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df["Model"], fontsize=10)
    ax.set_xlabel("R²"); ax.set_title("🧬 Model Comparison — Telomere Prediction", fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_comparison.png")


def save_results(results_df, svr_per_method):
    results_df.to_csv(f"{config.OUTPUT_DIR}/day16_results.csv", index=False, float_format='%.4f')
    pd.DataFrame([{"Method": m, **d} for m, d in svr_per_method.items()]).to_csv(
        f"{config.OUTPUT_DIR}/day16_selection_comparison.csv", index=False, float_format='%.4f')

    with open(f"{config.OUTPUT_DIR}/day16_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 16: TELOMERE LENGTH PREDICTION — FINAL REPORT\n" + "=" * 70 + "\n\n")
        f.write("FOCUS: Feature selection techniques + SVR\n\n")
        f.write("-" * 70 + "\nFEATURE SELECTION COMPARISON (SVR on each subset)\n" + "-" * 70 + "\n\n")
        for m, d in svr_per_method.items():
            f.write(f"  {m:15s} → {d['n_features']:>2d} feats | RMSE={d['cv_rmse']:.4f} | R²={d['cv_r2']:.4f}\n")
        f.write("\n" + "-" * 70 + "\nFINAL RESULTS\n" + "-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "-" * 70 + "\nKEY TAKEAWAYS\n" + "-" * 70 + "\n\n")
        f.write("1. Feature selection removes noise → SVR performs BETTER with fewer features\n")
        f.write("2. Correlation filter is simple but effective for linear relationships\n")
        f.write("3. Mutual information captures nonlinear dependencies correlation misses\n")
        f.write("4. RFECV is the gold standard but slowest (wraps model training)\n")
        f.write("5. Lasso does selection + regression simultaneously\n")
        f.write("6. Variance threshold alone is weak — high variance ≠ predictive power\n")
        f.write("7. SVR with RBF kernel is especially hurt by noisy features (curse of dimensionality)\n")
        f.write("8. Age and telomerase activity are the strongest biological predictors\n")
    logger.info("Results saved")
