from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('post_office_success_best_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    # Prepare the feature array
    features = np.array([
        data['avg_age'], data['gender_ratio'], data['avg_income'],
        data['employment_rate'], data['farming_cycle']
    ]).reshape(1, -1)
    
    # Predict success probabilities
    probabilities = model.predict(features)[0]
    
    # Prepare the response
    response = {
        "SB": probabilities[0],
        "RD": probabilities[1],
        "TD": probabilities[2],
        "MIS": probabilities[3],
        "SCSS": probabilities[4],
        "PPF": probabilities[5],
        "SSA": probabilities[6],
        "NSC": probabilities[7],
        "KVP": probabilities[8],
        "Mahila_Samman": probabilities[9],
        "PM_CARES": probabilities[10]
    }
    
    return jsonify(response)

