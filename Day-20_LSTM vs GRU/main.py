"""DAY 20: Viral Load Forecasting -- LSTM vs GRU -- Phase 2 Finale!"""
import matplotlib
matplotlib.use('Agg')
import os, sys, time, logging, warnings, numpy as np, torch
warnings.filterwarnings("ignore")

import config
for d in [config.DATA_DIR, config.MODEL_DIR, config.PLOT_DIR, config.LOG_DIR, config.OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in [logging.FileHandler(f"{config.LOG_DIR}/day20.log", mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    logger.addHandler(h)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, prepare_sequences
from model_training import train_all_models, save_models
from evaluation import evaluate_all, save_results


def main():
    t0 = time.time()
    logging.info("=" * 60)
    logging.info("  DAY 20: VIRAL LOAD FORECASTING")
    logging.info("  LSTM vs GRU -- Phase 2 Finale!")
    logging.info("=" * 60)

    data = load_data()
    explore_data(data)
    X_train, X_test, y_train, y_test, scaler = prepare_sequences(data)

    results = train_all_models(X_train, y_train)
    save_models(results)

    df = evaluate_all(results, X_test, y_test, scaler)
    save_results(df)

    logging.info(f"\n{'='*60}")
    logging.info(f"  DAY 20 COMPLETE | {time.time()-t0:.1f}s | Device: {config.DEVICE}")
    logging.info(f"  PHASE 2 COMPLETE! Next: Phase 3 -- Medical Imaging!")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
