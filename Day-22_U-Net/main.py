"""DAY 22: Brain Tumor Segmentation -- U-Net -- First Segmentation Project!"""
import matplotlib
matplotlib.use('Agg')
import os, sys, time, logging, warnings, numpy as np, torch
warnings.filterwarnings("ignore")

import config
for d in [config.DATA_DIR, config.MODEL_DIR, config.PLOT_DIR, config.LOG_DIR, config.OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in [logging.FileHandler(f"{config.LOG_DIR}/day22.log", mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    logger.addHandler(h)

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.RANDOM_SEED)

from data_pipeline import load_data, explore_data, create_loaders
from model_training import UNet, train_unet
from evaluation import evaluate_model, save_results


def main():
    t0 = time.time()
    logging.info("=" * 60)
    logging.info("  DAY 22: BRAIN TUMOR SEGMENTATION")
    logging.info("  U-Net -- First Segmentation Model!")
    logging.info("  Phase 3: Deep Learning & Medical Imaging")
    logging.info("=" * 60)

    images, masks = load_data()
    explore_data(images, masks)
    train_ld, val_ld, test_ld = create_loaders(images, masks)
    del images, masks  # free memory

    model = UNet()
    model, history = train_unet(model, train_ld, val_ld)
    torch.save(model.state_dict(), f"{config.MODEL_DIR}/day22_unet.pth")

    results = evaluate_model(model, test_ld)
    save_results(results)

    logging.info(f"\n{'='*60}")
    logging.info(f"  DAY 22 COMPLETE | {time.time()-t0:.1f}s | Device: {config.DEVICE}")
    logging.info(f"  Dice (tumor): {results['Dice (tumor only)']:.4f}")
    logging.info(f"  IoU (tumor):  {results['IoU (tumor only)']:.4f}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
