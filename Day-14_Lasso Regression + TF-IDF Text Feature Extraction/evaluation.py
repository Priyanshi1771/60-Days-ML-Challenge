"""Day 14: Drug Response — Evaluation"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config

logger = logging.getLogger(__name__)


def evaluate_all(lasso_results, baselines, nn_model, X_test, y_test, tfidf):
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    results = []

    # Best Lasso
    best_lasso = min(lasso_results, key=lambda r: r["cv_rmse"])
    y_pred = best_lasso["model"].predict(X_test).astype(np.float32)
    results.append(_m(f"Lasso (α={best_lasso['alpha']})", y_test, y_pred))

    # Show what Lasso selected
    _show_lasso_selected(best_lasso["model"], tfidf)

    # Baselines
    for name, data in baselines.items():
        y_pred = data["model"].predict(X_test).astype(np.float32)
        results.append(_m(name, y_test, y_pred))

    # GPU NN
    nn_model.eval()
    with torch.no_grad():
        X_dense = torch.from_numpy(X_test.toarray().astype(np.float32)).to(config.DEVICE)
        with autocast(device_type=config.DEVICE, enabled=config.DEVICE == "cuda"):
            y_nn = nn_model(X_dense).squeeze(-1).cpu().numpy()
    results.append(_m("GPU Neural Net", y_test, y_nn))

    df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    logger.info(f"\n{'='*60}\nRANKINGS\n{'='*60}")
    for i, row in df.iterrows():
        m = ["🥇","🥈","🥉","  "][min(i, 3)]
        logger.info(f"  {m} {row['Model']:25s} | RMSE={row['RMSE']:.4f} | R²={row['R²']:.4f}")
    return df


def _m(name, y_true, y_pred):
    return {"Model": name,
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R²": r2_score(y_true, y_pred)}


def _show_lasso_selected(model, tfidf):
    """Show which words Lasso kept (non-zero) and their coefficients."""
    feature_names = list(tfidf.get_feature_names_out()) + config.NUMERIC_FEATURES
    coefs = model.coef_
    nonzero_mask = coefs != 0
    n_kept = nonzero_mask.sum()
    logger.info(f"\n  🔪 Lasso kept {n_kept}/{len(coefs)} features ({n_kept/len(coefs)*100:.1f}%)")

    # Top positive (predict higher rating)
    top_pos_idx = np.argsort(coefs)[::-1][:10]
    logger.info(f"  📈 Top POSITIVE words (push rating UP):")
    for i in top_pos_idx:
        if coefs[i] > 0 and i < len(feature_names):
            logger.info(f"      {feature_names[i]:25s} coef = +{coefs[i]:.4f}")

    # Top negative (predict lower rating)
    top_neg_idx = np.argsort(coefs)[:10]
    logger.info(f"  📉 Top NEGATIVE words (push rating DOWN):")
    for i in top_neg_idx:
        if coefs[i] < 0 and i < len(feature_names):
            logger.info(f"      {feature_names[i]:25s} coef = {coefs[i]:.4f}")


def plot_predictions(lasso_results, nn_model, X_test, y_test):
    best_lasso = min(lasso_results, key=lambda r: r["cv_rmse"])
    y_lasso = best_lasso["model"].predict(X_test).astype(np.float32)

    nn_model.eval()
    with torch.no_grad():
        y_nn = nn_model(torch.from_numpy(X_test.toarray().astype(np.float32)).to(config.DEVICE)).squeeze(-1).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_pred, name, color in [
        (axes[0], y_lasso, f"Lasso (R²={r2_score(y_test, y_lasso):.3f})", '#AB47BC'),
        (axes[1], y_nn, f"GPU NN (R²={r2_score(y_test, y_nn):.3f})", '#4FC3F7')]:
        ax.scatter(y_test, y_pred, alpha=0.2, s=12, color=color, rasterized=True)
        ax.plot([1, 10], [1, 10], 'r--', lw=2)
        ax.set_xlabel("Actual Rating"); ax.set_ylabel("Predicted")
        ax.set_title(f"💊 {name}", fontweight='bold')
        ax.set_xlim(0.5, 10.5); ax.set_ylim(0.5, 10.5)
        ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_predictions.png")


def plot_comparison(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#AB47BC', '#4FC3F7', '#66BB6A', '#FF7043'][:len(results_df)]
    ax.barh(range(len(results_df)), results_df["R²"], color=colors, edgecolor='white')
    for i, row in results_df.iterrows():
        ax.text(max(0, row["R²"]) + 0.01, i, f'R²={row["R²"]:.4f} | RMSE={row["RMSE"]:.3f}', va='center', fontsize=9)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df["Model"], fontsize=10)
    ax.set_xlabel("R²"); ax.set_title("💊 Model Comparison — Drug Rating Prediction", fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/06_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 06_comparison.png")


def save_results(results_df):
    results_df.to_csv(f"{config.OUTPUT_DIR}/day14_results.csv", index=False, float_format='%.4f')
    with open(f"{config.OUTPUT_DIR}/day14_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 14: DRUG RESPONSE PREDICTION — FINAL REPORT\n" + "=" * 70 + "\n\n")
        f.write("FIRST TEXT+REGRESSION PROJECT: TF-IDF text features → predict drug rating\n\n")
        f.write("-" * 70 + "\nRESULTS\n" + "-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        f.write("\n\n" + "-" * 70 + "\nKEY TAKEAWAYS\n" + "-" * 70 + "\n\n")
        f.write("1. TF-IDF converts raw text into numerical features for ML models\n")
        f.write("2. Lasso (L1) automatically selects important words, zeroes out noise\n")
        f.write("3. With 5000 TF-IDF features, most are irrelevant → Lasso is ideal\n")
        f.write("4. Positive words ('excellent','relief') get positive coefficients\n")
        f.write("5. Negative words ('terrible','nausea') get negative coefficients\n")
        f.write("6. Bigrams (2-word phrases) capture context ('no side' vs 'side effects')\n")
        f.write("7. Neural net handles nonlinear text-rating relationships\n")
        f.write("8. Sparse matrix storage is critical — 5000 features × 6000 docs is huge\n")
    logger.info("Results saved")
