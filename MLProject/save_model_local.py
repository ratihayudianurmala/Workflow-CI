import os
import scipy.sparse as sp
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

# Set ke lokal
mlflow.set_tracking_uri("file:///C:/SMSML_Ratih Ayudia Nurmala/Workflow-CI/MLProject/mlruns")

X_train = sp.load_npz('olist_preprocessing/X_train.npz')
X_test = sp.load_npz('olist_preprocessing/X_test.npz')
y_train = pd.read_csv('olist_preprocessing/y_train.csv').squeeze()
y_test = pd.read_csv('olist_preprocessing/y_test.csv').squeeze()

with mlflow.start_run() as run:
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, name="model")
    print(f"Run ID: {run.info.run_id}")
    print("Model saved locally!")