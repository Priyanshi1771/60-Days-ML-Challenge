"""
DAY 12: Blood Pressure Prediction
❤️ Ridge Regression + Multicollinearity Handling + GPU Neural Net
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
fh = logging.FileHandler(f"{config.LOG_DIR}/day12_experiment.log", mode='w')
fh.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(ch)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, compute_vif, preprocess_and_split
from model_training import train_regression_showdown, train_gpu_nn, save_models
from evaluation import evaluate_all, plot_predictions, plot_regularization_path, save_results


def main():
    t0 = time.time()
    logging.info("╔" + "═" * 58 + "╗")
    logging.info("║  ❤️ DAY 12: BLOOD PRESSURE PREDICTION                    ║")
    logging.info("║  📉 Ridge/Lasso + Multicollinearity + GPU NN             ║")
    logging.info("╚" + "═" * 58 + "╝")

    X, y_sys, y_dia = load_data()
    explore_data(X, y_sys, y_dia)
    vifs = compute_vif(X)
    X_train, X_test, ys_train, ys_test, yd_train, yd_test, scaler = preprocess_and_split(X, y_sys, y_dia)

    # Regression showdown for both targets
    sys_results = train_regression_showdown(X_train, ys_train, "Systolic")
    dia_results = train_regression_showdown(X_train, yd_train, "Diastolic")

    # GPU neural net (dual output)
    nn_model = train_gpu_nn(X_train, ys_train, yd_train)

    save_models(sys_results, dia_results, nn_model, scaler)

    # Evaluate
    results_df, nn_pred = evaluate_all(sys_results, dia_results, nn_model, X_test, ys_test, yd_test)
    plot_predictions(sys_results, nn_pred, X_test, ys_test, yd_test)
    plot_regularization_path(X_train, ys_train)
    save_results(results_df)

    logging.info(f"\n{'='*60}")
    logging.info(f"❤️ DAY 12 COMPLETE | {time.time()-t0:.1f}s | Device: {config.DEVICE}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
