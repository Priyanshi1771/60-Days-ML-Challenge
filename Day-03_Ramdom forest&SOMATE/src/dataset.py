import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import SEED, TEST_SIZE

def load_and_preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Handle biological zeros (replace with NaN so the imputer can process them)
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    X[cols_with_zeros] = X[cols_with_zeros].replace(0, np.nan)
    
    return X, y

def create_splits(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)