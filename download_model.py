import mlflow
import os

os.environ['MLFLOW_TRACKING_USERNAME'] = 'ratihayudianurmala'
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('MLFLOW_TRACKING_PASSWORD', '')
mlflow.set_tracking_uri('https://dagshub.com/ratihayudianurmala/Eksperimen_SML_Ran.mlflow')

run_id = open('MLProject/last_run_id.txt').read().strip()
print(f"Run ID: {run_id}")
print(f"Username: {os.environ.get('MLFLOW_TRACKING_USERNAME')}")
print(f"Password length: {len(os.environ.get('MLFLOW_TRACKING_PASSWORD', ''))}")

client = mlflow.tracking.MlflowClient()
client.download_artifacts(run_id, 'model', './downloaded_model')
print('Model downloaded!')