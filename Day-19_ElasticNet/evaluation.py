"""Day 19: Radiosensitivity — Evaluation"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config

logger = logging.getLogger(__name__)


def evaluate_all(en_results, nn_model, splits):
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    rows = []
    for strat, data in en_results.items():
        rows.append({"Model": f"ElasticNet ({strat})", "R2": data["test_r2"],
                      "RMSE": data["test_rmse"],
                      "MAE": mean_absolute_error(data["y_true"], data["y_pred"])})

    # NN on within_A test
    X_te, y_te = splits["within_A"][1], splits["within_A"][3]
    nn_model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_te).to(config.DEVICE)
        with autocast(device_type=config.DEVICE, enabled=config.DEVICE == "cuda"):
            y_nn = nn_model(X_t).squeeze(-1).cpu().numpy()
    rows.append({"Model": "GPU Neural Net (within_A)", "R2": r2_score(y_te, y_nn),
                  "RMSE": np.sqrt(mean_squared_error(y_te, y_nn)),
                  "MAE": mean_absolute_error(y_te, y_nn)})

    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    logger.info(f"\n{'='*60}\nRANKINGS\n{'='*60}")
    for i, row in df.iterrows():
        logger.info(f"  {row['Model']:35s} | R2={row['R2']:.4f} | RMSE={row['RMSE']:.4f}")

    # Final comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4FC3F7', '#66BB6A', '#EF5350', '#7E57C2'][:len(df)]
    ax.barh(range(len(df)), df["R2"], color=colors, edgecolor='white')
    for i, row in df.iterrows():
        ax.text(max(0, row["R2"]) + 0.005, i,
                f'R2={row["R2"]:.4f} | RMSE={row["RMSE"]:.4f}', va='center', fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Model"], fontsize=8)
    ax.set_xlabel("R2"); ax.set_title("Model Comparison -- Radiosensitivity", fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_comparison.png")
    return df


def save_results(df):
    df.to_csv(f"{config.OUTPUT_DIR}/day19_results.csv", index=False, float_format='%.4f')
    with open(f"{config.OUTPUT_DIR}/day19_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 19: RADIOSENSITIVITY PREDICTION\n" + "=" * 70 + "\n\n")
        f.write("FOCUS: Elastic Net + Cross-Dataset Validation\n\n")
        f.write(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\nKEY TAKEAWAYS:\n")
        f.write("1. Cross-dataset performance always drops due to domain shift\n")
        f.write("2. Elastic Net balances L1 selection + L2 stability\n")
        f.write("3. Cell line models may not generalize to clinical tumors\n")
        f.write("4. l1_ratio controls the L1/L2 balance\n")
    logger.info("Results saved")
