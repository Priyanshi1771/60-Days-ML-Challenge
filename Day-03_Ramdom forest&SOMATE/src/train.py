import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.config import SEED, N_ESTIMATORS, MAX_DEPTH, MODELS_DIR

def build_and_train_pipeline(X_train, y_train):
    pipeline = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=SEED)),
        ('classifier', RandomForestClassifier(
            n_estimators=N_ESTIMATORS, 
            max_depth=MAX_DEPTH, 
            random_state=SEED, 
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # --- NEW: Save the model ---
    model_path = os.path.join(MODELS_DIR, 'rf_smote_pipeline.joblib')
    joblib.dump(pipeline, model_path)
    
    return pipeline