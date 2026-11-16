"""Web service that predicts the probability of heart disease for a patient."""

import pickle

from flask import Flask, request, jsonify

MODEL_FILE = 'model.bin'

CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('heart-disease')


@app.route('/ping', methods=['GET'])
def ping():
    return 'PONG'


@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    # categorical features were one-hot encoded as strings during training
    patient = {k: (str(v) if k in CATEGORICAL else v) for k, v in patient.items()}

    X = dv.transform([patient])
    probability = float(model.predict_proba(X)[0, 1])

    result = {
        'heart_disease_probability': round(probability, 3),
        'heart_disease': bool(probability >= 0.5),
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
