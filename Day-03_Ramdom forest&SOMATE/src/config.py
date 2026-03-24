import os

# Force the base directory to be exactly where this config.py file lives, minus one level (the src folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define exact paths
DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Create directories and explicitly print where they are
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=========================================")
print(f"📁 EXPECTED MODELS FOLDER: {MODELS_DIR}")
print(f"📁 EXPECTED OUTPUTS FOLDER: {OUTPUTS_DIR}")
print(f"📁 EXPECTED PLOTS FOLDER: {PLOTS_DIR}")
print("=========================================")

# Model Parameters
SEED = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 10