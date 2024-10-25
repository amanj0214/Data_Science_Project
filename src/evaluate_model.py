import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from data_preprocessing import preprocess_data, load_data  # Import the preprocessing functions


def evaluate_model():
    """
    Load the saved model and evaluate it on the preprocessed test data.
    """
    # Load the saved model and scaler
    model = joblib.load("./../models/best_model_final_full_data.joblib")
    scaler = joblib.load("./../models/scaler.joblib")

    # Load and preprocess the data (use the same scaler from training)
    data = load_data()
    X, y, _ = preprocess_data(data, fit_scaler=False, scaler=scaler)  # Apply the same scaler

    # Make predictions on the preprocessed test data
    y_pred = model.predict(X)

    # Calculate accuracy on the test data
    accuracy = accuracy_score(y, y_pred)
    print(f"Model Accuracy on full dataset: {accuracy}")

    return accuracy


if __name__ == '__main__':
    evaluate_model()
