import pandas as pd
from sqlalchemy import text

def load_data(engine):
    query = "SELECT * FROM titanic_raw_data"
    with engine.connect() as conn:
        data = pd.read_sql(text(query), conn)
    return data

def preprocess_data(data):
    # Simple preprocessing: drop NaNs, encode categorical features
    data = data.dropna()
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    return data

def save_processed_data(data, engine):
    data.to_sql('titanic_processed_data', con=engine, if_exists='replace', index=False)