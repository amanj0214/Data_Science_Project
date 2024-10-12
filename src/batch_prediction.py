import pandas as pd
from sqlalchemy import text
import model_repository

def batch_predict(engine):
    model = model_repository.load_model("titanic_rf_model")
    
    query = "SELECT * FROM titanic_raw_data WHERE prediction_made IS NULL"
    data = pd.read_sql(text(query), engine)
    
    X = preprocess_data(data)  # Use the same preprocessing
    predictions = model.predict(X)
    
    data['prediction'] = predictions
    data.to_sql('titanic_predictions', con=engine, if_exists='append', index=False)

    print("Batch prediction completed!")
