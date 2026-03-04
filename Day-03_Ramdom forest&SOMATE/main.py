import logging
import os
from src.config import DATA_PATH, MODELS_DIR
from src.dataset import load_and_preprocess_data, create_splits
from src.train import build_and_train_pipeline
from src.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == "__main__":
    logging.info("Starting Day 3 Pipeline: Diabetes Onset Prediction")
    
    # Mock data fallback
    if not os.path.exists(DATA_PATH):
        logging.warning("Data file not found. Generating mock data...")
        from sklearn.datasets import make_classification
        import pandas as pd
        X_mock, y_mock = make_classification(n_samples=768, n_features=8, weights=[0.65, 0.35], random_state=42)
        X = pd.DataFrame(X_mock, columns=['Preg', 'Gluc', 'BP', 'Skin', 'Ins', 'BMI', 'DPF', 'Age'])
        y = pd.Series(y_mock)
    else:
        logging.info("Loading real dataset...")
        X, y = load_and_preprocess_data(DATA_PATH)

    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = create_splits(X, y)
    
    logging.info("Training pipeline and saving model to models/ directory...")
    model_pipeline = build_and_train_pipeline(X_train, y_train)
    logging.info(f"Model saved successfully to {MODELS_DIR}.")
    
    logging.info("Evaluating model, saving metrics, and generating plots...")
    # Pass X.columns so the plot knows the names of the features
    evaluate_model(model_pipeline, X_test, y_test, feature_names=X.columns)
    
    logging.info("Run Complete. Check the models, outputs, and plots folders!")