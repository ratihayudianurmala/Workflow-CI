import mlflow
import os

os.environ['MLFLOW_TRACKING_USERNAME'] = 'ratihayudianurmala'
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('MLFLOW_TRACKING_PASSWORD', '')
mlflow.set_tracking_uri('https://dagshub.com/ratihayudianurmala/Eksperimen_SML_Ran.mlflow')

run_id = open('MLProject/last_run_id.txt').read().strip()
print(f"Downloading model for run: {run_id}")
mlflow.artifacts.download_artifacts(f'runs:/{run_id}/model', dst_path='./downloaded_model')
print('Model downloaded!')