"""Day 18: Gene Expression — Evaluation"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config

logger = logging.getLogger(__name__)


def evaluate_all(arch_results, baselines, X_test_np, y_test_np, X_test_t, y_test_t):
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST SET")
    logger.info("=" * 60)

    results = []

    # DL models
    for name, data in arch_results.items():
        model = data["model"]
        model.eval()
        with torch.no_grad():
            with autocast(device_type=config.DEVICE, enabled=config.DEVICE == "cuda"):
                y_pred = model(X_test_t).squeeze(-1).cpu().numpy()
        results.append(_m(name, y_test_np, y_pred, data["n_params"]))

    # Sklearn baselines
    for name, model in baselines.items():
        y_pred = model.predict(X_test_np).astype(np.float32)
        n_p = getattr(model, 'n_features_in_', 0) * 2  # rough param estimate
        results.append(_m(name, y_test_np, y_pred, n_p))

    df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    logger.info(f"\n{'='*60}\nRANKINGS\n{'='*60}")
    for i, row in df.iterrows():
        m = ["1st", "2nd", "3rd"][min(i, 2)] if i < 3 else f"{i+1}th"
        tag = "DL" if row["Params"] > 1000 else "ML"
        logger.info(f"  {m:4s} [{tag}] {row['Model']:22s} | RMSE={row['RMSE']:.4f} | R2={row['R2']:.4f}")

    return df


def _m(name, y_true, y_pred, n_params):
    return {"Model": name, "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred), "Params": n_params}


def plot_best_predictions(arch_results, baselines, X_test_np, y_test_np, X_test_t):
    """Actual vs predicted for best DL model and best ML model."""
    # Best DL
    best_dl_name = min(arch_results, key=lambda k: arch_results[k]["rmse"])
    best_dl = arch_results[best_dl_name]["model"]
    best_dl.eval()
    with torch.no_grad():
        y_dl = best_dl(X_test_t).squeeze(-1).cpu().numpy()

    # Best ML (Ridge usually)
    y_ridge = baselines["Ridge"].predict(X_test_np)
    y_rf = baselines["Random Forest"].predict(X_test_np)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # DL prediction
    axes[0, 0].scatter(y_test_np, y_dl, alpha=0.3, s=12, color='#7E57C2', rasterized=True)
    axes[0, 0].plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', lw=2)
    r2_dl = r2_score(y_test_np, y_dl)
    axes[0, 0].set_title(f"DL: {best_dl_name} (R2={r2_dl:.4f})", fontweight='bold')
    axes[0, 0].set_xlabel("Actual"); axes[0, 0].set_ylabel("Predicted")
    axes[0, 0].spines[['top', 'right']].set_visible(False)

    # Ridge prediction
    axes[0, 1].scatter(y_test_np, y_ridge, alpha=0.3, s=12, color='#4FC3F7', rasterized=True)
    axes[0, 1].plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', lw=2)
    r2_r = r2_score(y_test_np, y_ridge)
    axes[0, 1].set_title(f"Ridge Regression (R2={r2_r:.4f})", fontweight='bold')
    axes[0, 1].set_xlabel("Actual"); axes[0, 1].set_ylabel("Predicted")
    axes[0, 1].spines[['top', 'right']].set_visible(False)

    # DL residuals
    resid_dl = y_test_np - y_dl
    axes[1, 0].scatter(y_dl, resid_dl, alpha=0.3, s=12, color='#FF7043', rasterized=True)
    axes[1, 0].axhline(0, color='k', lw=1, linestyle='--')
    axes[1, 0].set_title("DL Residuals", fontweight='bold')
    axes[1, 0].set_xlabel("Predicted"); axes[1, 0].set_ylabel("Residual")
    axes[1, 0].spines[['top', 'right']].set_visible(False)

    # Error distribution comparison
    axes[1, 1].hist(np.abs(y_test_np - y_dl), bins=30, alpha=0.6, color='#7E57C2', label='Best DL', edgecolor='white')
    axes[1, 1].hist(np.abs(y_test_np - y_ridge), bins=30, alpha=0.6, color='#4FC3F7', label='Ridge', edgecolor='white')
    axes[1, 1].hist(np.abs(y_test_np - y_rf), bins=30, alpha=0.6, color='#66BB6A', label='RF', edgecolor='white')
    axes[1, 1].set_title("Absolute Error Distribution", fontweight='bold')
    axes[1, 1].set_xlabel("|Error|"); axes[1, 1].legend()
    axes[1, 1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_predictions.png")


def plot_dl_vs_ml(results_df):
    """Bar chart comparing all models."""
    fig, ax = plt.subplots(figsize=(10, 7))
    n = len(results_df)

    # Color by type (DL vs ML)
    colors = ['#7E57C2' if row['Params'] > 1000 else '#4FC3F7' for _, row in results_df.iterrows()]

    ax.barh(range(n), results_df["R2"], color=colors, edgecolor='white')
    for i, row in results_df.iterrows():
        tag = "DL" if row["Params"] > 1000 else "ML"
        ax.text(max(0, row["R2"]) + 0.005, i,
                f'R2={row["R2"]:.4f} | RMSE={row["RMSE"]:.4f} [{tag}]', va='center', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(results_df["Model"], fontsize=9)
    ax.set_xlabel("R2")
    ax.set_title("DL vs ML -- Gene Expression Prediction\n(Purple=DL | Blue=ML)", fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_dl_vs_ml.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_dl_vs_ml.png")


def plot_capacity_vs_performance(arch_results):
    """Does deeper/wider always mean better?"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#7E57C2', '#4FC3F7', '#66BB6A', '#FF7043', '#EF5350']

    for i, (name, data) in enumerate(arch_results.items()):
        ax.scatter(data["n_params"], data["rmse"], s=200, color=colors[i],
                   edgecolors='white', linewidth=2, zorder=5)
        layers_str = "->".join(map(str, data["hidden"]))
        ax.annotate(f'{name.split("(")[0].strip()}\n[{layers_str}]',
                    (data["n_params"], data["rmse"]), fontsize=7, ha='center', va='top',
                    xytext=(0, -15), textcoords='offset points')

    ax.set_xlabel("# Trainable Parameters (model capacity)", fontsize=11)
    ax.set_ylabel("Validation RMSE (lower = better)", fontsize=11)
    ax.set_title("Capacity vs Performance -- Is Bigger Always Better?\n"
                 "(Optimal = bottom-left corner)", fontweight='bold')
    ax.grid(alpha=0.3); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_capacity_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_capacity_curve.png")


def save_results(results_df, arch_results):
    results_df.to_csv(f"{config.OUTPUT_DIR}/day18_results.csv", index=False, float_format='%.4f')

    arch_df = pd.DataFrame([
        {"Architecture": n, "Hidden": str(d["hidden"]), "Params": d["n_params"],
         "Val_RMSE": d["rmse"], "Epochs": d["epochs_trained"], "Time_s": d["time"]}
        for n, d in arch_results.items()
    ])
    arch_df.to_csv(f"{config.OUTPUT_DIR}/day18_architectures.csv", index=False, float_format='%.4f')

    with open(f"{config.OUTPUT_DIR}/day18_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 18: GENE EXPRESSION PREDICTION -- FINAL REPORT\n" + "=" * 70 + "\n\n")
        f.write("FIRST PURE DL REGRESSION PROJECT\n")
        f.write("FOCUS: Architecture comparison on high-dimensional genomic data\n\n")
        f.write("-" * 70 + "\nARCHITECTURE COMPARISON\n" + "-" * 70 + "\n\n")
        f.write(arch_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "-" * 70 + "\nFINAL RESULTS (DL + ML)\n" + "-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "-" * 70 + "\nKEY TAKEAWAYS\n" + "-" * 70 + "\n\n")
        f.write("1. DL can capture nonlinear gene interactions that linear models miss\n")
        f.write("2. Deeper is NOT always better -- medium (2-layer) often wins on small data\n")
        f.write("3. Bottleneck architectures learn compressed representations (like autoencoders)\n")
        f.write("4. Wide networks risk overfitting on limited genomic samples\n")
        f.write("5. 500 input features with 350 noise genes -- DL must learn to ignore noise\n")
        f.write("6. Batch normalization is critical for deep genomic networks\n")
        f.write("7. Dropout prevents co-adaptation -- essential when features >> samples\n")
        f.write("8. This is the FOUNDATION for Days 36+ (RNA folding, multi-omics, etc.)\n")
    logger.info("Results saved")
