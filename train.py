"""Train the final heart disease model and save it to model.bin."""

import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

C = 0.1
DATA_PATH = 'data/heart.csv'
OUTPUT_FILE = 'model.bin'

NUMERICAL = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
FEATURES = CATEGORICAL + NUMERICAL


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)

    # invalid codes in the original data are missing values: replace with the most frequent value
    df.loc[df['ca'] == 4, 'ca'] = df['ca'].mode()[0]
    df.loc[df['thal'] == 0, 'thal'] = df['thal'].mode()[0]

    for c in CATEGORICAL:
        df[c] = df[c].astype(str)
    return df


def train(df, C=C):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(df[FEATURES].to_dict(orient='records'))
    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X, df['target'].values)
    return dv, model


def predict(df, dv, model):
    X = dv.transform(df[FEATURES].to_dict(orient='records'))
    return model.predict_proba(X)[:, 1]


if __name__ == '__main__':
    df = load_and_clean(DATA_PATH)

    # same split as in the notebook: evaluate on the test set
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    dv, model = train(df_full_train)
    y_pred = predict(df_test, dv, model)
    y_test = df_test['target'].values
    print(f'Test AUC      = {roc_auc_score(y_test, y_pred):.3f}')
    print(f'Test accuracy = {accuracy_score(y_test, y_pred >= 0.5):.3f}')

    # final model: trained on the whole dataset
    dv, model = train(df)
    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f'Model saved to {OUTPUT_FILE}')
