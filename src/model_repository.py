import joblib
import os

def save_model(model, model_name):
    model_dir = os.getenv('MODEL_REPO_DIR', 'models/')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)

def load_model(model_name):
    model_dir = os.getenv('MODEL_REPO_DIR', 'models/')
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    return joblib.load(model_path)
