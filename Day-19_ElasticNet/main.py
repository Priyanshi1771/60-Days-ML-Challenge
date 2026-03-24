"""DAY 19: Radiosensitivity Prediction -- Elastic Net + Cross-Dataset Validation"""
import matplotlib
matplotlib.use('Agg')
import os, sys, time, logging, warnings, numpy as np, torch
warnings.filterwarnings("ignore")

import config
for d in [config.DATA_DIR, config.MODEL_DIR, config.PLOT_DIR, config.LOG_DIR, config.OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in [logging.FileHandler(f"{config.LOG_DIR}/day19.log", mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    logger.addHandler(h)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

from data_pipeline import load_datasets, explore_data, prepare_splits
from model_training import train_elastic_net_all_strategies, train_gpu_nn, save_models
from evaluation import evaluate_all, save_results


def main():
    t0 = time.time()
    logging.info("=" * 60)
    logging.info("  DAY 19: RADIOSENSITIVITY PREDICTION")
    logging.info("  Elastic Net + Cross-Dataset Validation + GPU NN")
    logging.info("=" * 60)

    X_a, y_a, X_b, y_b = load_datasets()
    explore_data(X_a, y_a, X_b, y_b)
    splits, scaler = prepare_splits(X_a, y_a, X_b, y_b)

    en_results = train_elastic_net_all_strategies(splits)
    nn_model = train_gpu_nn(splits["within_A"][0], splits["within_A"][2])
    save_models(en_results, nn_model, scaler)

    df = evaluate_all(en_results, nn_model, splits)
    save_results(df)

    logging.info(f"\n{'='*60}")
    logging.info(f"  DAY 19 COMPLETE | {time.time()-t0:.1f}s | Device: {config.DEVICE}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
