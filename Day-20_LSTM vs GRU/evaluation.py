"""Day 20: Viral Load — Evaluation"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config

logger = logging.getLogger(__name__)


def evaluate_all(results, X_test, y_test, scaler):
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    X_t = torch.from_numpy(X_test).to(config.DEVICE)
    rows = []
    preds = {}

    for name, data in results.items():
        model = data["model"]; model.eval()
        with torch.no_grad():
            with autocast(device_type=config.DEVICE, enabled=config.DEVICE == "cuda"):
                y_pred_s = model(X_t).cpu().numpy()

        y_pred = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
        y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        preds[name] = y_pred

        rows.append({"Model": name,
                      "MAE": mean_absolute_error(y_actual, y_pred),
                      "RMSE": np.sqrt(mean_squared_error(y_actual, y_pred)),
                      "R2": r2_score(y_actual, y_pred)})
        logger.info(f"  {name:6s} | RMSE={rows[-1]['RMSE']:.4f} | R2={rows[-1]['R2']:.4f}")

    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Plots
    best_name = df.iloc[0]["Model"]
    y_best = preds[best_name]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    n_show = min(200, len(y_actual))
    axes[0].plot(y_actual[:n_show], 'k-', lw=1.5, label='Actual', alpha=0.8)
    axes[0].plot(y_best[:n_show], '--', color='#7E57C2', lw=1.5, label=f'{best_name}')
    axes[0].set_title(f"Forecast ({best_name})", fontweight='bold')
    axes[0].set_xlabel("Sample"); axes[0].set_ylabel("log10(copies/mL)")
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].scatter(y_actual, y_best, alpha=0.2, s=8, color='#7E57C2', rasterized=True)
    axes[1].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
    axes[1].set_title(f"Actual vs Predicted (R2={r2_score(y_actual, y_best):.4f})", fontweight='bold')
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    axes[1].spines[['top', 'right']].set_visible(False)

    axes[2].hist(y_actual - y_best, bins=40, color='#FF7043', edgecolor='white', alpha=0.8)
    axes[2].axvline(0, color='k', lw=1.5, linestyle='--')
    axes[2].set_title("Error Distribution", fontweight='bold')
    axes[2].set_xlabel("Error"); axes[2].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_results.png")
    return df


def save_results(df):
    df.to_csv(f"{config.OUTPUT_DIR}/day20_results.csv", index=False, float_format='%.4f')
    with open(f"{config.OUTPUT_DIR}/day20_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 20: VIRAL LOAD FORECASTING\n" + "=" * 70 + "\n\n")
        f.write("FOCUS: LSTM vs GRU sequential modeling on GPU\n\n")
        f.write(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\nKEY TAKEAWAYS:\n")
        f.write("1. LSTM and GRU both capture treatment response decay patterns\n")
        f.write("2. GRU has fewer params but often matches LSTM\n")
        f.write("3. Viral rebound is the hardest pattern to predict\n")
        f.write("4. Phase 2 complete! 10/10 regression projects done!\n")
    logger.info("Results saved")
