from flask import Flask, request, render_template, jsonify
# import src.model_repository as model_repository
import pandas as pd

app = Flask(__name__)

# Load model using the abstracted load_model function
# model = model_repository.load_model("titanic_survival_model")

# Example lists for dropdown options
states = ["CA", "NY", "TX"]
item_classes = ["Electronics", "Furniture", "Clothing"]
item_categories = ["A", "B", "C"]

@app.route('/')
def index():
    return render_template('index.html', states=states, item_classes=item_classes, item_categories=item_categories)


@app.route('/get_lagged_return_rate', methods=['POST'])
def get_lagged_return_rate():
    # Get the selected state from the request
    selected_state = request.json.get("state")

    # Calculate the lagged return rate based on the selected state
    # (You can replace this with your actual calculation logic)
    lagged_return_rate = {
        "CA": 0.12,
        "NY": 0.15,
        "TX": 0.10
    }.get(selected_state, 0.0)

    return jsonify({"lagged_return_rate": lagged_return_rate})


@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({"predicted_return_rate": 45})
    # Assume that form data contains the input features in JSON
    data = request.get_json(force=True)
    return jsonify({"predicted_return_rate": 45})
    df = pd.DataFrame([data])
    
    # Make predictions using the loaded model
    prediction = model.predict(df)
    
    # Send the prediction result back to the client
    return jsonify({"predicted_return_rate": int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=False)
