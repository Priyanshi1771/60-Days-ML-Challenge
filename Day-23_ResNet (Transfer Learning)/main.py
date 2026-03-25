"""DAY 23: Skin Lesion Classification -- ResNet Transfer Learning"""
import matplotlib
matplotlib.use('Agg')
import os, sys, time, logging, warnings, numpy as np, torch
warnings.filterwarnings("ignore")

import config
for d in [config.DATA_DIR, config.MODEL_DIR, config.PLOT_DIR, config.LOG_DIR, config.OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in [logging.FileHandler(f"{config.LOG_DIR}/day23.log", mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    logger.addHandler(h)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.RANDOM_SEED)

from data_pipeline import load_data, create_loaders, plot_samples
from model_training import train_all_strategies
from evaluation import evaluate_all_strategies, save_results


def main():
    t0 = time.time()
    logging.info("=" * 60)
    logging.info("  DAY 23: SKIN LESION CLASSIFICATION")
    logging.info("  ResNet18 Transfer Learning -- 3 Strategies Compared")
    logging.info("  First Transfer Learning Project!")
    logging.info("=" * 60)

    images, labels = load_data()
    plot_samples(images, labels)
    train_ld, val_ld, test_ld = create_loaders(images, labels)
    del images  # free memory

    results = train_all_strategies(train_ld, val_ld)

    # Save best model
    best_name = max(results, key=lambda k: results[k]["best_acc"])
    torch.save(results[best_name]["model"].state_dict(), f"{config.MODEL_DIR}/day23_best.pth")

    df, _ = evaluate_all_strategies(results, test_ld)
    save_results(df)

    logging.info(f"\n{'='*60}")
    logging.info(f"  DAY 23 COMPLETE | {time.time()-t0:.1f}s | Device: {config.DEVICE}")
    logging.info(f"  Best strategy: {df.iloc[0]['Strategy']} | Acc: {df.iloc[0]['Accuracy']:.4f}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
