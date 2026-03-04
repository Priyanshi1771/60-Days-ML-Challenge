import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
from src.config import OUTPUTS_DIR, PLOTS_DIR

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(importances, feature_names, save_path):
    plt.figure(figsize=(8, 6))
    # Sort features by importance
    indices = importances.argsort()[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Gini Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(pipeline, X_test, y_test, feature_names):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 1. Generate and Save Classification Report
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    report_path = os.path.join(OUTPUTS_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Clinical Evaluation Report\n")
        f.write("==========================\n")
        f.write(report)
        f.write(f"\nROC-AUC Score: {auc:.4f}\n")
    
    print(f"Metrics saved to {report_path}")
    
    # 2. Generate and Save Plots
    cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, cm_path)
    
    roc_path = os.path.join(PLOTS_DIR, 'roc_curve.png')
    plt.figure(figsize=(6, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title('ROC Curve')
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    
    # Extract feature importances
    rf_step = pipeline.named_steps['classifier']
    feat_imp_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
    plot_feature_importance(rf_step.feature_importances_, feature_names, feat_imp_path)
    
    print(f"Plots saved to {PLOTS_DIR}/")