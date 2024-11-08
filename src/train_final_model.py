import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import mlflow
import joblib

from utils.data_processing import load_data, preprocess_data

def main():
    # Load and preprocess data
    data = load_data("data/tpcds.db")
    processed_data = preprocess_data(data)

    # Define features and target
    X = processed_data.drop(columns=['return_rate'])
    y = processed_data['return_rate']

    # Load best parameters and preprocessor from cross-validation
    best_params = joblib.load("models/best_params.pkl")
    best_params = {k.replace("model__", ""): v for k, v in best_params.items()}
    preprocessor = joblib.load("models/preprocessor.pkl")

    # Train final model pipeline
    final_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(objective='reg:squarederror', **best_params))
    ])

    with mlflow.start_run():
        final_model.fit(X, y)
        mlflow.sklearn.log_model(final_model, "final_model")
        print("Final model trained and saved.")

if __name__ == "__main__":
    main()
