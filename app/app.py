import setup_src_path

from flask import Flask, request, render_template, jsonify
import pandas as pd
import predict
import db_utils

app = Flask(__name__)

# Example lists for dropdown options
states = db_utils.get_all_states()
item_classes = db_utils.get_all_item_class()
item_categories = db_utils.get_all_item_categories()

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
def predict_return_rate():
    data = request.get_json(force=True)
    prediction_return_rate = predict.predict(pd.DataFrame([data]))
    return jsonify({"predicted_return_rate": float(prediction_return_rate[0])})


if __name__ == '__main__':
    app.run(debug=False)
