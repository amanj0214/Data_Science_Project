import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from sqlalchemy import text
from models.model_repository import save_model

def train_model(engine):
    query = "SELECT * FROM titanic_processed_data"
    data = pd.read_sql(text(query), engine)
    
    X = data.drop(columns=['Survived'])
    y = data['Survived']
    
    # Step 1: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Define models and parameters to search over
    model_params = {
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100],
                'criterion': ['gini', 'entropy']
            }
        },
        'SVC': {
            'model': SVC(),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        }
    }
    
    best_model = None
    best_params = None
    best_score = 0
    
    # Step 2: Loop through each model and parameter set
    for model_name, mp in model_params.items():
        grid = GridSearchCV(mp['model'], mp['params'], cv=5)
        grid.fit(X_train, y_train)
        
        # Log parameters and results in MLflow
        with mlflow.start_run(run_name=f"{model_name}_GridSearch"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("best_score", grid.best_score_)
            
            # Check if this is the best model so far
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_model = grid.best_estimator_
                best_params = grid.best_params_
    
    # Log best model and parameters
    with mlflow.start_run(run_name="Best_Model_Retrain"):
        # Retrain the best model on the full dataset (train + test)
        best_model.fit(X, y)
        
        # Log retrained model
        mlflow.sklearn.log_model(best_model, "best_model_retrained")
        mlflow.log_params(best_params)
        mlflow.log_metric("retrained_on_full_data", True)
        
        # Save model for production use
        save_model(best_model, "titanic_best_model_final")
    
    return best_model, X_test, y_test
