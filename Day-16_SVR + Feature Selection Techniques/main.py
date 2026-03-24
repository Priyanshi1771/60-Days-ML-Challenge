"""
DAY 16: Telomere Length Prediction
🧬 SVR + Feature Selection Techniques + GPU Neural Net
"""
import matplotlib
matplotlib.use('Agg')
import os, sys, time, logging, warnings, numpy as np, torch
warnings.filterwarnings("ignore")

import config
for d in [config.DATA_DIR, config.MODEL_DIR, config.PLOT_DIR, config.LOG_DIR, config.OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in [logging.FileHandler(f"{config.LOG_DIR}/day16_experiment.log", mode='w'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    logger.addHandler(h)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, preprocess_and_split, run_feature_selection_comparison
from model_training import train_svr_per_selection, train_best_svr, train_baselines, train_gpu_nn, save_models
from evaluation import evaluate_all, plot_predictions, plot_comparison, save_results


def main():
    t0 = time.time()
    logging.info("╔" + "═" * 58 + "╗")
    logging.info("║  🧬 DAY 16: TELOMERE LENGTH PREDICTION                   ║")
    logging.info("║  🔬 SVR + Feature Selection Deep-Dive                    ║")
    logging.info("║  ⚡ GPU Neural Net Comparison                            ║")
    logging.info("╚" + "═" * 58 + "╝")

    X, y = load_data()
    explore_data(X, y)
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(X, y)

    # Feature selection comparison (6 methods)
    selection_results = run_feature_selection_comparison(X_train, y_train, X_test)

    # Train SVR on each feature subset
    svr_per_method = train_svr_per_selection(X_train, y_train, selection_results)

    # Find best method
    best_method = min(svr_per_method, key=lambda m: svr_per_method[m]["cv_rmse"])
    best_mask = selection_results[best_method]["mask"]
    logging.info(f"\n  🏆 Best selection method: {best_method} ({best_mask.sum()} features)")

    # Full GridSearch SVR on best features
    svr_model, grid = train_best_svr(X_train, y_train, best_mask)
    baselines = train_baselines(X_train, y_train, best_mask)
    nn_model = train_gpu_nn(X_train, y_train, best_mask)
    save_models(svr_model, baselines, nn_model, scaler, best_mask)

    # Evaluate
    results_df, preds_dict = evaluate_all(svr_model, baselines, nn_model, X_test, y_test, best_mask)
    plot_predictions(y_test, preds_dict, X_test)
    plot_comparison(results_df)
    save_results(results_df, svr_per_method)

    logging.info(f"\n{'='*60}")
    logging.info(f"🧬 DAY 16 COMPLETE | {time.time()-t0:.1f}s | Device: {config.DEVICE}")
    logging.info(f"  Best selection: {best_method} | Best model: {results_df.iloc[0]['Model']}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
