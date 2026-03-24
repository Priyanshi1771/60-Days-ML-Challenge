import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.config import OUTPUTS_DIR, PLOTS_DIR

def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 1. Save Report
    report_path = os.path.join(OUTPUTS_DIR, 'classification_report.txt')
    print(f"💾 Attempting to save report to: {report_path}")
    with open(report_path, 'w') as f:
        f.write("Clinical Evaluation Report: Liver Cirrhosis (SVM)\n")
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}\n")
    print("✅ Report saved!")

    # 2. Save Confusion Matrix
    cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    print(f"🖼️ Attempting to save Confusion Matrix plot to: {cm_path}")
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Oranges', cbar=False)
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    
    # 3. Save ROC Curve
    roc_path = os.path.join(PLOTS_DIR, 'roc_curve.png')
    print(f"🖼️ Attempting to save ROC Curve plot to: {roc_path}")
    plt.figure(figsize=(6, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, color='darkorange')
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # 4. Save Decision Boundary
    boundary_path = os.path.join(PLOTS_DIR, 'svm_decision_boundary.png')
    print(f"🖼️ Attempting to save Decision Boundary plot to: {boundary_path}")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(StandardScaler().fit_transform(X_train))
    svm_2d = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
    svm_2d.fit(X_train_pca, y_train)

    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=20)
    plt.tight_layout()
    plt.savefig(boundary_path)
    plt.close()
    print("✅ All plots saved!")