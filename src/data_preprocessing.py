import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

def load_data():
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    data = pd.concat([X, y], axis=1)
    return data

def preprocess_data(data, fit_scaler=True, scaler=None):
    # Simple preprocessing: drop NaNs, encode categorical features
    data = data[['survived', 'pclass', 'sex']].copy()
    data['sex'] = data['sex'].map({'male': 0, 'female': 1}).astype(int)
    data = data.dropna().copy()

    # Separate features and labels
    X = data.drop(columns=['survived'])
    y = data['survived']

    if fit_scaler:
        # Fit a new scaler during training
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        # Use the provided scaler for inference
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=scaler.get_feature_names_out())

    return X_scaled, y, scaler  # Ensure returning scaler, even if not fitted

def save_processed_data(data):
    data.to_csv('./../data/processed/processed.csv', index=False)
    print(f"File Wrote {data.shape}")

def main():
    data = load_data()
    X_scaled, y, _ = preprocess_data(data)
    save_processed_data(pd.concat([X_scaled, y], axis=1))

if __name__ == '__main__':
    main()
