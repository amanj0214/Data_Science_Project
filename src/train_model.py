import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from data_preprocessing import preprocess_data, load_data  # Import from your data_preprocessing module


def train_model():
    # Load and preprocess data
    data = load_data()
    X, y, scaler = preprocess_data(data, fit_scaler=True)  # Fit scaler during training

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and return the best model using GridSearchCV
    best_model = train_sklearn_with_gridsearch(X_train, X_test, y_train, y_test, X, y)

    # Save the trained model and the scaler
    joblib.dump(best_model, './../models/best_model_final_full_data.joblib')
    joblib.dump(scaler, './../models/scaler.joblib')
    print("Best model and scaler trained and saved.")

    return best_model


def train_sklearn_with_gridsearch(X_train, X_test, y_train, y_test, X, y):
    # Define model parameters for GridSearch
    model_params = {
        'RandomForestClassifier1': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100],
                'criterion': ['gini', 'entropy']
            }
        },
        'RandomForestClassifier2': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [200, 150],
            }
        }
    }

    best_model = None
    best_params = None
    best_score = 0

    # Scikit-learn GridSearchCV
    for model_name, mp in model_params.items():
        grid = GridSearchCV(mp['model'], mp['params'], cv=5)
        grid.fit(X_train, y_train)

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_params = grid.best_params_

    print(f"Best Model Params: {best_params}")
    print(f"Best Cross-Validation Score: {best_score}")

    # Evaluate the model on the test set
    evaluate_model(best_model, X_test, y_test)

    # Retrain the best model on the full data
    best_model.fit(X, y)

    return best_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and print accuracy.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {accuracy}")
    return accuracy


if __name__ == '__main__':
    train_model()
