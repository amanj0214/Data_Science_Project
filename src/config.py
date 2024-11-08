import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed_data.csv')
MODEL_FILE = os.path.join(MODELS_DIR, 'final_model.pkl')
MLFLOW_TRACKING_URI = "mlruns"
