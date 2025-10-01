import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv('data/train.tsv', sep='\t')
X_train, X_test, y_train, y_test = train_test_split(
    df['sentence'], df['target'], test_size=0.2, random_state=42
)

pipeline = Pipeline([
  ("tfidf", TfidfVectorizer()),
  ("clf", RandomForestClassifier(random_state=93))
])

pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)

y_pred_proba = pipeline.predict_proba(X_test)
# print(confusion_matrix(y_test, y_pred))

import mlflow
import os
os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:8080'

mlflow.set_experiment("Thresholding Experiment")

import numpy as np
thresholds = np.linspace(0, 1, 10)
for t in thresholds:
  with mlflow.start_run():
    y_pred = (y_pred_proba[:, 1] >= t).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_param("threshold", t)
    mlflow.log_metric("precision", report['1']['precision'])
    mlflow.log_metric("recall", report['1']['recall'])
