from src.config import DATA_PATH
from src.dataset import load_and_preprocess_data, create_splits
from src.train import build_and_train_pipeline
from src.evaluate import evaluate_model

def run_end_to_end_pipeline():
    print("\n--- 🚀 STARTING ML PIPELINE ---")
    X, y = load_and_preprocess_data(DATA_PATH)
    X_train, X_test, y_train, y_test = create_splits(X, y)
    
    print("\n--- 🧠 TRAINING MODEL ---")
    model_pipeline = build_and_train_pipeline(X_train, y_train)
    
    print("\n--- 📊 EVALUATING AND SAVING ARTIFACTS ---")
    evaluate_model(model_pipeline, X_train, y_train, X_test, y_test)
    
    print("\n--- 🎉 PIPELINE COMPLETE ---")