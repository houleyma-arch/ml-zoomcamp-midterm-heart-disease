"""Send one example patient to the running web service."""

import requests

url = 'http://localhost:9696/predict'

patient = {
    'age': 54,
    'sex': 1,
    'cp': 2,
    'trestbps': 130,
    'chol': 246,
    'fbs': 0,
    'restecg': 1,
    'thalach': 173,
    'exang': 0,
    'oldpeak': 0.0,
    'slope': 2,
    'ca': 0,
    'thal': 2,
}

response = requests.post(url, json=patient)
print(response.json())
