import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
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

    # Cross-validation strategy
    tscv = TimeSeriesSplit(n_splits=5)

    # Define preprocessing pipeline
    categorical_cols = ['ca_state', 'i_class', 'i_category']
    numeric_cols = ['lag_1_return_rate']
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(drop='first'), categorical_cols),
        ("numerical", StandardScaler(), numeric_cols)
    ])

    # Define model pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(objective='reg:squarederror'))
    ])

    # Define hyperparameter grid
    param_grid = {
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.1],
        'model__n_estimators': [500]
    }

    # Cross-validation with grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring=make_scorer(mean_squared_error, greater_is_better=False)
    )

    with mlflow.start_run():
        # Run grid search
        grid_search.fit(X, y)
        
        # Log best parameters
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)
        
        # Save the best parameters and preprocessor pipeline
        joblib.dump(best_params, "models/best_params.pkl")
        joblib.dump(preprocessor, "models/preprocessor.pkl")
        print("Best parameters and preprocessor saved:", best_params)

        # Log metrics
        best_model = grid_search.best_estimator_
        mse = mean_squared_error(y, best_model.predict(X))
        mlflow.log_metric("MSE", mse)
        print("Cross-validation MSE:", mse)

if __name__ == "__main__":
    main()
