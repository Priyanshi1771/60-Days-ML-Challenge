"""
DAY 21: Pneumonia Detection
CNN + Data Augmentation -- Phase 3 Begins! (Deep Learning & Medical Imaging)
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
for h in [logging.FileHandler(f"{config.LOG_DIR}/day21.log", mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    logger.addHandler(h)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

from data_pipeline import load_data, create_loaders, plot_samples
from model_training import PneumoniaCNN, train_model
from evaluation import evaluate_model


def main():
    t0 = time.time()
    logging.info("=" * 60)
    logging.info("  DAY 21: PNEUMONIA DETECTION")
    logging.info("  4-Block CNN + Data Augmentation")
    logging.info("  Phase 3 Begins: Deep Learning & Medical Imaging!")
    logging.info("=" * 60)

    # Data
    images, labels = load_data()
    plot_samples(images, labels)
    train_loader, val_loader, test_loader = create_loaders(images, labels)
    del images  # free memory

    # Train
    model = PneumoniaCNN()
    model, history = train_model(model, train_loader, val_loader)

    # Save model
    torch.save(model.state_dict(), f"{config.MODEL_DIR}/day21_cnn.pth")
    logging.info(f"  Model saved: {config.MODEL_DIR}/day21_cnn.pth")

    # Evaluate
    results = evaluate_model(model, test_loader)

    elapsed = time.time() - t0
    logging.info(f"\n{'='*60}")
    logging.info(f"  DAY 21 COMPLETE | {elapsed:.1f}s | Device: {config.DEVICE}")
    logging.info(f"  Accuracy: {results['Accuracy']:.4f} | F1: {results['F1']:.4f} | AUC: {results['AUC-ROC']:.4f}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
