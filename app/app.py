from flask import Flask, request, render_template, jsonify
# import src.model_repository as model_repository
import pandas as pd

app = Flask(__name__)

# Load model using the abstracted load_model function
# model = model_repository.load_model("titanic_survival_model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({"prediction": 45})
    # Assume that form data contains the input features in JSON
    data = request.get_json(force=True)
    return jsonify({"prediction": 45})
    df = pd.DataFrame([data])
    
    # Make predictions using the loaded model
    prediction = model.predict(df)
    
    # Send the prediction result back to the client
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=False)
