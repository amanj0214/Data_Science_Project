import pandas as pd
import joblib
from data_preprocessing import preprocess_data  # Reuse your existing preprocessing
from sklearn.model_selection import train_test_split


def load_model():
    """
    Load the saved model and preprocessing pipeline.
    """
    model = joblib.load("./../models/best_model_final_full_data.joblib")
    print("Model loaded for batch predictions.")
    return model


def load_new_data(file_path):
    """
    Load new data from a CSV or other data source.
    The data should have the same features as the training data used to train the model.
    """
    data = pd.read_csv(file_path)
    return data


def make_batch_predictions(model, data, batch_size=100):
    """
    Make predictions on new data in batches to avoid memory issues with large datasets.
    """
    num_rows = data.shape[0]
    predictions = []

    # Loop through the data in batches
    for i in range(0, num_rows, batch_size):
        batch_data = data[i:i + batch_size]
        batch_predictions = model.predict(batch_data)
        predictions.extend(batch_predictions)

    return predictions


def save_predictions(predictions, output_file):
    """
    Save the predictions to a CSV file or database.
    """
    predictions_df = pd.DataFrame(predictions, columns=['predicted_survived'])
    predictions_df.to_csv(output_file, index=False)
    print(f"Batch predictions saved to {output_file}")


if __name__ == '__main__':
    # Load the model
    model = load_model()

    # Load new data for predictions
    new_data = load_new_data("./../data/new/new_data.csv")

    # Apply the preprocessing pipeline
    preprocessed_data = preprocess_data(new_data)

    # Make batch predictions
    predictions = make_batch_predictions(model, preprocessed_data)

    # Save predictions to a file
    save_predictions(predictions, './../data/predictions/batch_predictions.csv')
