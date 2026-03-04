import os
import joblib
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.config import SEED, SVM_C, SVM_GAMMA, MODELS_DIR

def build_and_train_pipeline(X_train, y_train):
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', C=SVM_C, gamma=SVM_GAMMA, 
                           class_weight='balanced', probability=True, random_state=SEED))
    ])
    
    pipeline.fit(X_train, y_train)
    
    model_path = os.path.join(MODELS_DIR, 'svm_liver_pipeline.joblib')
    print(f"💾 Attempting to save model to: {model_path}")
    joblib.dump(pipeline, model_path)
    if os.path.exists(model_path):
        print("✅ Model successfully saved!")
    else:
        print("❌ ERROR: Model failed to save.")
        
    return pipeline