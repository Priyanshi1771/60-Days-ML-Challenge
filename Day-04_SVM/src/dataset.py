import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import SEED, TEST_SIZE

def load_and_preprocess_data(filepath):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        target_col = df.columns[-1]
        df[target_col] = df[target_col].apply(lambda x: 1 if x == 1 else 0)
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        if 'Gender' in X.columns:
            from sklearn.preprocessing import LabelEncoder
            X['Gender'] = LabelEncoder().fit_transform(X['Gender'])
    else:
        print(f"⚠️ Warning: Dataset not found at {filepath}. Generating mock clinical data...")
        from sklearn.datasets import make_classification
        X_mock, y_mock = make_classification(n_samples=583, n_features=10, n_informative=6, weights=[0.29, 0.71], random_state=SEED)
        feature_names = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'AG_Ratio']
        X = pd.DataFrame(X_mock, columns=feature_names)
        X['Gender'] = np.where(X['Gender'] > 0, 1, 0)
        y = pd.Series(y_mock)

    return X, y

def create_splits(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)