# Heart Disease Prediction

Midterm project for the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalks.Club.

## Problem description

Heart disease is one of the leading causes of death worldwide. Diagnosing it usually needs several
clinical tests and a specialist's assessment. A model that estimates the risk of heart disease from
routine measurements can help doctors **prioritise patients**: patients with a high predicted risk
can be sent for further examination first, while low risk patients can be followed up normally.

In this project, I train a **binary classification** model that predicts whether a patient has
heart disease (`target = 1`) or not (`target = 0`) from 13 clinical features, such as age, chest pain
type, resting blood pressure, cholesterol, maximum heart rate and exercise test results.

The model is served as a **web service**: a client (for example a hospital application) sends the
patient's measurements as JSON and receives the probability of heart disease.

> This is an educational project. The model is not a medical device and must not be used for real diagnosis.

## Dataset

UCI Heart Disease dataset (Cleveland), in the version published on Kaggle:
[Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci).
It contains 303 patients and 14 columns. The dataset is committed in this repository: `data/heart.csv`.

| column | meaning |
|---|---|
| age | age in years |
| sex | 1 = male, 0 = female |
| cp | chest pain type (0 to 3) |
| trestbps | resting blood pressure (mm Hg) |
| chol | serum cholesterol (mg/dl) |
| fbs | fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | resting electrocardiographic results (0 to 2) |
| thalach | maximum heart rate achieved |
| exang | exercise induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise relative to rest |
| slope | slope of the peak exercise ST segment (0 to 2) |
| ca | number of major vessels colored by fluoroscopy (0 to 3) |
| thal | thalassemia test result (1 to 3) |
| target | 1 = heart disease, 0 = no heart disease |

## What is in the notebook

`notebook.ipynb` contains:

* **Data cleaning:** removal of one duplicated row; invalid codes (`ca = 4`, `thal = 0`, which are missing values in the original data) replaced by the most frequent value
* **EDA:** target distribution (54% / 46%, fairly balanced), ranges and distributions of numerical features, box plots by target, correlation matrix, disease rate per category
* **Feature importance:** mutual information for categorical features, correlation with the target for numerical features. The most important features are `thal`, `cp`, `ca`, `exang`, `oldpeak` and `thalach`
* **Model training and tuning** (metric: ROC AUC), with a 60% / 20% / 20% train / validation / test split:
  * Logistic Regression (tuning `C`)
  * Decision Tree (tuning `max_depth`, `min_samples_leaf`)
  * Random Forest (tuning `n_estimators`, `max_depth`, `min_samples_leaf`)
  * XGBoost (tuning `eta`, `max_depth`, `min_child_weight`)
  * 5-fold cross-validation to compare the best models

### Results

| model | validation AUC | 5-fold CV AUC |
|---|---|---|
| Logistic Regression (C = 0.1) | 0.936 | 0.916 |
| Random Forest | 0.936 | 0.914 |
| XGBoost | 0.932 | 0.909 |

The scores are very close, so I chose **Logistic Regression (C = 0.1)**: it is the simplest and
most interpretable model, and it needs no extra library in the web service.

Final model on the test set: **AUC = 0.900, accuracy = 0.836**.

## Project structure

```
.
├── data/heart.csv            # dataset
├── notebook.ipynb            # cleaning, EDA, feature importance, model selection
├── train.py                  # trains the final model and saves it to model.bin
├── model.bin                 # trained model (DictVectorizer + LogisticRegression)
├── predict.py                # Flask web service
├── test_predict.py           # sends an example patient to the service
├── requirements.txt          # all dependencies (notebook + service)
├── requirements-service.txt  # dependencies of the web service only (used by Docker)
└── Dockerfile
```

## How to run the project

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ml-zoomcamp-midterm-heart-disease.git
cd ml-zoomcamp-midterm-heart-disease
```

### 2. Create and activate a virtual environment, install the dependencies

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. (Optional) Run the notebook

```bash
jupyter notebook notebook.ipynb
```

### 4. Train the model

```bash
python train.py
```

Output:

```
Test AUC      = 0.900
Test accuracy = 0.836
Model saved to model.bin
```

### 5. Run the web service locally

```bash
gunicorn --bind=0.0.0.0:9696 predict:app
```

(On Windows, gunicorn does not work: use `python predict.py` instead.)

In a second terminal (with the virtual environment activated):

```bash
python test_predict.py
```

Expected answer:

```
{'heart_disease': True, 'heart_disease_probability': 0.942}
```

You can also use `curl`:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 0, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
       "thalach": 120, "exang": 1, "oldpeak": 3.0, "slope": 0, "ca": 2, "thal": 3}'
```

```
{"heart_disease": false, "heart_disease_probability": 0.025}
```

## Run with Docker

Build the image (the model is trained during the build):

```bash
docker build -t heart-disease .
```

Run the container:

```bash
docker run -it --rm -p 9696:9696 heart-disease
```

Then, in another terminal, test it:

```bash
python test_predict.py
```

## Tip: run everything without installing anything

You can run the whole project in **GitHub Codespaces** (Python and Docker are already installed):
on the repository page click **Code > Codespaces > Create codespace on main**, then follow the steps above in the terminal.
