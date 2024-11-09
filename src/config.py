from pathlib import Path

BASE_DIR = Path(__file__)
SQL_LITE_FILE_NAME = BASE_DIR / '../../data/raw/tpcds.db'
PROCESSED_DATA_FILE = BASE_DIR / '../../data/processed/processed.csv'
LATEST_MODEL_FILE = BASE_DIR / '../../models/latest/final_model.pkl'
MODEL_VERSIONS_FOLDER = BASE_DIR / '../../models/versioned_models/'
MLFLOW_TRACKING_URI = "mlruns"
