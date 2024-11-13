import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import mlflow
import joblib
from load_data import load_data
from data_processing import process_data
import config


def main():
    # Load and preprocess data
    data = load_data()
    processed_data = process_data(data)

    model_columns = ['d_year', 'd_moy', 'ca_state', 'i_class', 'i_category',
                     'return_rate', 'lag_1_return_rate']

    # Define Features and Target
    X = processed_data[model_columns].drop(columns=['return_rate'])
    y = processed_data['return_rate']

    # Loadbest parameters and preprocessor from cross-validation
    best_params = joblib.load(config.BEST_PARAMS_FILE)
    best_params = {k.replace("model__", ""): v for k, v in best_params.items()}
    preprocessor = joblib.load(config.PREPROCESSOR_OBJ_FILE)

    # Train final model pipeline
    final_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(objective='reg:squarederror', **best_params))
    ])

    with mlflow.start_run():
        final_model.fit(X, y)
        mlflow.sklearn.log_model(final_model, "final_model")
        joblib.dump(final_model, config.LATEST_MODEL_FILE)
        print("Final model trained and saved.")


if __name__ == "__main__":
    main()
