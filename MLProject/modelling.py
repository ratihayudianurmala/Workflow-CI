import os
import mlflow
import mlflow.sklearn
import scipy.sparse as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['MLFLOW_TRACKING_USERNAME'] = 'ratihayudianurmala'
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('MLFLOW_TRACKING_PASSWORD', '')
mlflow.set_tracking_uri('https://dagshub.com/ratihayudianurmala/Eksperimen_SML_Ran.mlflow')

run_id = open('MLProject/last_run_id.txt').read().strip()
print(f"Downloading model for run: {run_id}")
mlflow.artifacts.download_artifacts(f'runs:/{run_id}/model', dst_path='./downloaded_model')
print('Model downloaded!')

# Load data
X_train = sp.load_npz('olist_preprocessing/X_train.npz')
X_test = sp.load_npz('olist_preprocessing/X_test.npz')
y_train = pd.read_csv('olist_preprocessing/y_train.csv').squeeze()
y_test = pd.read_csv('olist_preprocessing/y_test.csv').squeeze()

with mlflow.start_run(run_name="logistic-regression-baseline") as run:

    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", auc)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negatif', 'Positif'],
                yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    report = classification_report(y_test, y_pred)
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    mlflow.log_artifact('classification_report.txt')

    mlflow.sklearn.log_model(model, artifact_path="model")

    # Simpan run ID untuk Docker build
    with open('last_run_id.txt', 'w') as f:
        f.write(run.info.run_id)

    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")
    print("Model dan artefak berhasil di-log ke DagsHub!")