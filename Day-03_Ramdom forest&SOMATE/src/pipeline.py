import os
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.datasets import make_classification

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ==========================================
# 1. CONFIGURATION & FOLDER SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not BASE_DIR:  # Fallback if run in some interactive environments
    BASE_DIR = os.getcwd()

MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Force create the directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

SEED = 42

# ==========================================
# 2. DATA PREPARATION (Mock Data for instant testing)
# ==========================================
logging.info("Generating mock medical data...")
X_mock, y_mock = make_classification(n_samples=768, n_features=8, weights=[0.65, 0.35], random_state=SEED)
X = pd.DataFrame(X_mock, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age'])
y = pd.Series(y_mock)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# ==========================================
# 3. PIPELINE DEFINITION & TRAINING
# ==========================================
logging.info("Building and training SMOTE -> Random Forest pipeline...")
pipeline = ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=SEED)),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1))
])

pipeline.fit(X_train, y_train)

# SAVE THE MODEL
model_path = os.path.join(MODELS_DIR, 'rf_smote_pipeline.joblib')
joblib.dump(pipeline, model_path)
logging.info(f"Model saved to: {model_path}")

# ==========================================
# 4. EVALUATION & PLOTTING
# ==========================================
logging.info("Evaluating model and generating plots...")
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# SAVE TEXT REPORT
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

report_path = os.path.join(OUTPUTS_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("Clinical Evaluation Report\n==========================\n")
    f.write(report)
    f.write(f"\nROC-AUC Score: {auc:.4f}\n")
logging.info(f"Report saved to: {report_path}")

# SAVE CONFUSION MATRIX PLOT
cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(cm_path)
plt.close()

# SAVE ROC CURVE PLOT
roc_path = os.path.join(PLOTS_DIR, 'roc_curve.png')
plt.figure(figsize=(6, 4))
RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
plt.title('ROC Curve')
plt.tight_layout()
plt.savefig(roc_path)
plt.close()

# SAVE FEATURE IMPORTANCE PLOT
feat_imp_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
importances = pipeline.named_steps['classifier'].feature_importances_
indices = importances.argsort()[::-1]
sorted_features = [X.columns[i] for i in indices]
sorted_importances = importances[indices]

plt.figure(figsize=(8, 6))
sns.barplot(x=sorted_importances, y=sorted_features, hue=sorted_features, legend=False, palette='viridis')
plt.title('Random Forest Feature Importance')
plt.xlabel('Gini Importance')
plt.tight_layout()
plt.savefig(feat_imp_path)
plt.close()

logging.info(f"Plots saved to: {PLOTS_DIR}")
logging.info("PIPELINE COMPLETE! Check your folders now.")