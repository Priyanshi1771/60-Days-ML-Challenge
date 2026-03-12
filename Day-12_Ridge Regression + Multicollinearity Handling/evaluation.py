"""Day 12: Blood Pressure — Evaluation"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config

logger = logging.getLogger(__name__)


def evaluate_all(sys_results, dia_results, nn_model, X_test, ys_test, yd_test):
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)

    all_results = []

    # sklearn models
    for target, results, y_true, label in [
        ("Systolic", sys_results, ys_test, "Sys"),
        ("Diastolic", dia_results, yd_test, "Dia")
    ]:
        for name, data in results.items():
            y_pred = data["model"].predict(X_test).astype(np.float32)
            all_results.append(_metrics(f"{name} ({label})", y_true, y_pred))

    # GPU NN (dual output)
    nn_model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).to(config.DEVICE)
        with autocast(device_type=config.DEVICE, enabled=config.DEVICE == "cuda"):
            pred_both = nn_model(X_t).cpu().numpy()
    all_results.append(_metrics("GPU NN (Sys)", ys_test, pred_both[:, 0]))
    all_results.append(_metrics("GPU NN (Dia)", yd_test, pred_both[:, 1]))

    df = pd.DataFrame(all_results).sort_values("RMSE").reset_index(drop=True)

    logger.info(f"\n{'='*60}\nRESULTS\n{'='*60}")
    for _, row in df.iterrows():
        logger.info(f"  {row['Model']:28s} | RMSE={row['RMSE']:.3f} | MAE={row['MAE']:.3f} | R²={row['R²']:.4f}")

    return df, pred_both


def _metrics(name, y_true, y_pred):
    return {"Model": name,
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R²": r2_score(y_true, y_pred)}


def plot_predictions(sys_results, nn_pred, X_test, ys_test, yd_test):
    """Actual vs Predicted + Residuals + Bland-Altman for systolic."""
    # Best sklearn
    best_name = min(sys_results, key=lambda k: sys_results[k]["cv_rmse"])
    y_sklearn = sys_results[best_name]["model"].predict(X_test)
    y_nn_sys = nn_pred[:, 0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: Actual vs Predicted
    for ax, y_pred, name, color in [
        (axes[0,0], y_sklearn, f"Best sklearn ({best_name})", '#42A5F5'),
        (axes[0,1], y_nn_sys, "GPU Neural Net", '#66BB6A'),
    ]:
        ax.scatter(ys_test, y_pred, alpha=0.2, s=10, color=color, rasterized=True)
        ax.plot([80,200],[80,200], 'r--', lw=2)
        r2 = r2_score(ys_test, y_pred)
        ax.set_title(f"❤️ {name}\nR²={r2:.4f}", fontweight='bold', fontsize=11)
        ax.set_xlabel("Actual Systolic"); ax.set_ylabel("Predicted")
        ax.spines[['top','right']].set_visible(False)

    # Bland-Altman plot (clinical standard for comparing methods)
    mean_vals = (ys_test + y_nn_sys) / 2
    diff_vals = ys_test - y_nn_sys
    mean_diff = diff_vals.mean()
    std_diff = diff_vals.std()
    axes[0,2].scatter(mean_vals, diff_vals, alpha=0.2, s=10, color='#AB47BC', rasterized=True)
    axes[0,2].axhline(mean_diff, color='red', lw=2, label=f'Mean diff={mean_diff:.2f}')
    axes[0,2].axhline(mean_diff + 1.96*std_diff, color='orange', ls='--', lw=1.5, label='±1.96 SD')
    axes[0,2].axhline(mean_diff - 1.96*std_diff, color='orange', ls='--', lw=1.5)
    axes[0,2].set_title("📐 Bland-Altman Plot (Clinical Standard)", fontweight='bold', fontsize=11)
    axes[0,2].set_xlabel("Mean of Actual & Predicted"); axes[0,2].set_ylabel("Actual − Predicted")
    axes[0,2].legend(fontsize=9); axes[0,2].spines[['top','right']].set_visible(False)

    # Row 2: Residuals
    for ax, y_pred, name, color in [
        (axes[1,0], y_sklearn, "sklearn Residuals", '#FF7043'),
        (axes[1,1], y_nn_sys, "NN Residuals", '#EF5350'),
    ]:
        residuals = ys_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.2, s=10, color=color, rasterized=True)
        ax.axhline(0, color='k', ls='--', lw=1)
        ax.set_title(f"📊 {name}", fontweight='bold', fontsize=11)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
        ax.spines[['top','right']].set_visible(False)

    # Error distribution
    axes[1,2].hist(ys_test - y_sklearn, bins=30, alpha=0.6, color='#42A5F5', label='sklearn', edgecolor='white')
    axes[1,2].hist(ys_test - y_nn_sys, bins=30, alpha=0.6, color='#66BB6A', label='GPU NN', edgecolor='white')
    axes[1,2].set_title("📊 Error Distribution", fontweight='bold')
    axes[1,2].set_xlabel("Prediction Error (mmHg)"); axes[1,2].legend()
    axes[1,2].spines[['top','right']].set_visible(False)

    plt.suptitle("Blood Pressure Prediction — Comprehensive Analysis", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_predictions.png")


def plot_regularization_path(X_train, y_train):
    """Show how Ridge alpha controls coefficient shrinkage."""
    from sklearn.linear_model import Ridge
    alphas = np.logspace(-3, 4, 50)
    coefs = []
    for a in alphas:
        r = Ridge(alpha=a); r.fit(X_train, y_train)
        coefs.append(r.coef_)
    coefs = np.array(coefs)

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, name in enumerate(config.FEATURE_NAMES):
        ax.plot(alphas, coefs[:, i], lw=1.5, label=name)
    ax.set_xscale('log')
    ax.axvline(x=1.0, color='red', ls='--', lw=2, alpha=0.5, label='α=1.0')
    ax.set_xlabel("Ridge α (log scale)", fontsize=11)
    ax.set_ylabel("Coefficient Value", fontsize=11)
    ax.set_title("📉 Ridge Regularization Path\n(Higher α → more shrinkage → more stable but biased)",
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(alpha=0.3); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/07_ridge_path.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 07_ridge_path.png")


def save_results(results_df):
    results_df.to_csv(f"{config.OUTPUT_DIR}/day12_results.csv", index=False, float_format='%.4f')
    with open(f"{config.OUTPUT_DIR}/day12_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 12: BLOOD PRESSURE PREDICTION — FINAL REPORT\n" + "=" * 70 + "\n\n")
        f.write("FOCUS: Multicollinearity detection + Ridge/Lasso regularization\n\n")
        f.write("-" * 70 + "\nRESULTS\n" + "-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "-" * 70 + "\nKEY TAKEAWAYS\n" + "-" * 70 + "\n\n")
        f.write("1. Multicollinearity inflates OLS coefficients — Ridge/Lasso fix this\n")
        f.write("2. VIF > 10 = severe collinearity (BMI↔weight, total↔LDL cholesterol)\n")
        f.write("3. Ridge shrinks ALL coefficients → stable but keeps all features\n")
        f.write("4. Lasso zeros out redundant features → automatic feature selection\n")
        f.write("5. ElasticNet combines L1+L2 → best of both worlds\n")
        f.write("6. GPU NN learns both systolic+diastolic in one forward pass\n")
        f.write("7. Bland-Altman plot is the clinical standard for method comparison\n")
    logger.info("Results saved")
