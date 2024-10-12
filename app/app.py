from flask import Flask, request, render_template, jsonify
from src.titanic_ml_pipeline.model_loader import load_model
import pandas as pd

app = Flask(__name__)

# Load model using the abstracted load_model function
model = load_model("titanic_survival_model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Assume that form data contains the input features in JSON
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    
    # Make predictions using the loaded model
    prediction = model.predict(df)
    
    # Send the prediction result back to the client
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
