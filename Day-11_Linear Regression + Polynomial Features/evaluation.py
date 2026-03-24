"""Day 11: ICU Mortality — Evaluation (Regression Metrics)"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config

logger = logging.getLogger(__name__)


def evaluate_all(poly_results, nn_model, X_test, y_test):
    """Evaluate polynomial models + neural net on test set."""
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)

    results = []

    # Polynomial models
    for degree, data in poly_results.items():
        y_pred = data["model"].predict(X_test).astype(np.float32)
        results.append(_compute_metrics(f"Poly Degree {degree} (Ridge)", y_test, y_pred))

    # GPU Neural Net
    nn_model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).to(config.DEVICE)
        with autocast(device_type=config.DEVICE, enabled=config.DEVICE == "cuda"):
            y_nn = nn_model(X_t).cpu().numpy()
    results.append(_compute_metrics("GPU Neural Net", y_test, y_nn))

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    logger.info(f"\n{'='*60}\nRANKINGS (by RMSE)\n{'='*60}")
    for i, row in results_df.iterrows():
        medal = ["🥇","🥈","🥉","  "][min(i, 3)]
        logger.info(f"  {medal} {row['Model']:28s} | RMSE={row['RMSE']:.4f} | R²={row['R²']:.4f} | MAE={row['MAE']:.4f}")

    return results_df


def _compute_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2,
            "MSE": mean_squared_error(y_true, y_pred)}


def plot_predictions(poly_results, nn_model, X_test, y_test):
    """Actual vs Predicted scatter + residual plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Best poly model
    best_deg = min(poly_results, key=lambda d: poly_results[d]["cv_rmse_mean"])
    y_poly = poly_results[best_deg]["model"].predict(X_test).astype(np.float32)

    # Neural net
    nn_model.eval()
    with torch.no_grad():
        y_nn = nn_model(torch.from_numpy(X_test).to(config.DEVICE)).cpu().numpy()

    # (0,0) Actual vs Predicted — best poly
    axes[0,0].scatter(y_test, y_poly, alpha=0.3, s=12, color='#4FC3F7', rasterized=True)
    axes[0,0].plot([0,1],[0,1], 'r--', lw=2, label='Perfect')
    r2 = r2_score(y_test, y_poly)
    axes[0,0].set_title(f"🏥 Poly({best_deg}) Ridge — R²={r2:.4f}", fontweight='bold')
    axes[0,0].set_xlabel("Actual"); axes[0,0].set_ylabel("Predicted")
    axes[0,0].legend(); axes[0,0].spines[['top','right']].set_visible(False)

    # (0,1) Actual vs Predicted — NN
    axes[0,1].scatter(y_test, y_nn, alpha=0.3, s=12, color='#66BB6A', rasterized=True)
    axes[0,1].plot([0,1],[0,1], 'r--', lw=2, label='Perfect')
    r2_nn = r2_score(y_test, y_nn)
    axes[0,1].set_title(f"🧠 GPU Neural Net — R²={r2_nn:.4f}", fontweight='bold')
    axes[0,1].set_xlabel("Actual"); axes[0,1].set_ylabel("Predicted")
    axes[0,1].legend(); axes[0,1].spines[['top','right']].set_visible(False)

    # (1,0) Residuals — poly
    residuals_poly = y_test - y_poly
    axes[1,0].scatter(y_poly, residuals_poly, alpha=0.3, s=12, color='#FF7043', rasterized=True)
    axes[1,0].axhline(0, color='k', linestyle='--', lw=1)
    axes[1,0].set_title(f"📊 Residuals — Poly({best_deg})", fontweight='bold')
    axes[1,0].set_xlabel("Predicted"); axes[1,0].set_ylabel("Residual")
    axes[1,0].spines[['top','right']].set_visible(False)

    # (1,1) Residuals — NN
    residuals_nn = y_test - y_nn
    axes[1,1].scatter(y_nn, residuals_nn, alpha=0.3, s=12, color='#AB47BC', rasterized=True)
    axes[1,1].axhline(0, color='k', linestyle='--', lw=1)
    axes[1,1].set_title("📊 Residuals — Neural Net", fontweight='bold')
    axes[1,1].set_xlabel("Predicted"); axes[1,1].set_ylabel("Residual")
    axes[1,1].spines[['top','right']].set_visible(False)

    plt.suptitle("Actual vs Predicted + Residual Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_predictions.png")


def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    colors = ['#4FC3F7', '#66BB6A', '#FF7043', '#AB47BC'][:len(results_df)]
    ax.barh(x, results_df["R²"], color=colors, alpha=0.85, edgecolor='white')
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax.text(row["R²"] + 0.005, i, f'R²={row["R²"]:.4f} | RMSE={row["RMSE"]:.4f}', va='center', fontsize=9)
    ax.set_yticks(x)
    ax.set_yticklabels(results_df["Model"], fontsize=10)
    ax.set_xlabel("R² Score")
    ax.set_title("🏥 Model Comparison — ICU Mortality Prediction", fontweight='bold', fontsize=13)
    ax.set_xlim(0, 1.0)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_comparison.png")


def plot_feature_importance(poly_results, feature_names):
    """Show polynomial regression coefficients for degree 1 (interpretable)."""
    model = poly_results[1]["model"]
    coefs = model.named_steps["reg"].coef_
    n_orig = len(feature_names)
    coefs = coefs[:n_orig]  # Only original features for degree 1

    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    top_n = min(15, len(sorted_idx))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#EF5350' if coefs[i] > 0 else '#4FC3F7' for i in sorted_idx[:top_n]]
    ax.barh(range(top_n), coefs[sorted_idx[:top_n]], color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]], fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', lw=1, linestyle='--')
    ax.set_xlabel("Coefficient Value")
    ax.set_title("🔬 Linear Regression Coefficients\n(Red = increases risk | Blue = decreases risk)",
                 fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_coefficients.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_coefficients.png")


def save_results(results_df):
    results_df.to_csv(f"{config.OUTPUT_DIR}/day11_results.csv", index=False, float_format='%.4f')
    with open(f"{config.OUTPUT_DIR}/day11_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 11: ICU MORTALITY PREDICTION — FINAL REPORT\n" + "=" * 70 + "\n\n")
        f.write("PHASE 2 BEGINS: Regression & Time-Series\n\n")
        f.write("OBJECTIVE: Predict ICU mortality risk score using regression\n")
        f.write("FOCUS: Polynomial features + regression metrics (MAE, RMSE, R²)\n\n")
        f.write("-" * 70 + "\nRESULTS\n" + "-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "-" * 70 + "\nKEY TAKEAWAYS\n" + "-" * 70 + "\n\n")
        f.write("1. Polynomial features capture nonlinear relationships in linear models\n")
        f.write("2. Degree 2 is usually the sweet spot — degree 3+ causes feature explosion\n")
        f.write("3. Ridge regularization is essential to prevent poly overfitting\n")
        f.write("4. RMSE penalizes large errors more than MAE\n")
        f.write("5. R² measures proportion of variance explained (1.0 = perfect)\n")
        f.write("6. Residual plots reveal systematic prediction errors\n")
        f.write("7. GPU neural net can capture arbitrary nonlinearities without manual features\n")
    logger.info("Results saved")
