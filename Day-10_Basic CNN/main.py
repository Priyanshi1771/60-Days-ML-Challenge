"""
=============================================================================
 DAY 10: Malaria Cell Classification — 🎉 ENTERING DEEP LEARNING!
=============================================================================
 🦟 OBJECTIVE: Classify malaria-infected vs healthy blood cells using CNN
 🧠 KEY LEARNING: First Convolutional Neural Network — built from scratch
 📊 DATASET: NIH Malaria Cell Images (or synthetic fallback)
 ⚡ MODEL: Custom 3-block CNN with BN, Dropout, AdaptiveAvgPool
 
 THIS IS THE MILESTONE:
   Days 1-9:  Classical ML (sklearn)
   Day 10:    🎉 DEEP LEARNING BEGINS (PyTorch)
   Days 11+:  Regression, Time-Series, Transfer Learning, U-Net...
   
 ▶️ USAGE: cd day10_malaria_classification && python main.py
=============================================================================
"""
import matplotlib
matplotlib.use('Agg')
import os, sys, time, logging, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")

import config
for d in [config.DATA_DIR, config.MODEL_DIR, config.PLOT_DIR, config.LOG_DIR, config.OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('training.log', encoding='utf-8')
fh = logging.FileHandler(f"{config.LOG_DIR}/day10_experiment.log", mode='w')
fh.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
logger.addHandler(ch)

# Reproducibility
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from data_pipeline import load_data, create_dataloaders, plot_sample_images
from model_training import MalariaCNN, train_model, save_model, count_parameters
from evaluation import (evaluate_model, plot_confusion_matrix, plot_roc_curve,
                         plot_confidence_distribution, visualize_feature_maps,
                         error_analysis, save_results)


def main():
    total_start = time.time()
    logging.info("╔" + "═" * 58 + "╗")
    logging.info("║  🦟 DAY 10: MALARIA CELL CLASSIFICATION                 ║")
    logging.info("║  🧠 Model: Custom CNN (First Deep Learning Project!)     ║")
    logging.info("║  📊 Dataset: Cell Images → Parasitized vs Uninfected     ║")
    logging.info("║  🎉 MILESTONE: Entering Deep Learning!                   ║")
    logging.info("╚" + "═" * 58 + "╝")

    # Phase 1: Data
    images, labels = load_data()
    plot_sample_images(images, labels)
    train_loader, val_loader, test_loader = create_dataloaders(images, labels)
    
    # Free raw images from memory after DataLoaders are created
    del images
    
    # Phase 2: Build + Train CNN
    model = MalariaCNN()
    logging.info(f"  CNN Parameters: {count_parameters(model):,}")
    model, history = train_model(model, train_loader, val_loader)
    save_model(model, history)

    # Phase 3: Evaluate
    results, y_pred, y_proba, y_test = evaluate_model(model, test_loader)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_confidence_distribution(y_test, y_proba)
    visualize_feature_maps(model, test_loader)
    error_analysis(y_test, y_pred, y_proba)
    save_results(results)

    total_time = time.time() - total_start
    logging.info(f"\n{'='*60}")
    logging.info("🦟 DAY 10 COMPLETE — Welcome to Deep Learning! 🎉")
    logging.info(f"{'='*60}")
    logging.info(f"  Runtime: {total_time:.1f}s")
    logging.info(f"  Accuracy: {results['Accuracy']:.4f} | AUC: {results['AUC-ROC']:.4f}")
    logging.info(f"  Device: {config.DEVICE} | Params: {count_parameters(model):,}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
