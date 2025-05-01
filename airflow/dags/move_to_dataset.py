import sqlite3
import os
import io
from PIL import Image
import subprocess
import requests

import sys
import importlib.util
import subprocess

if importlib.util.find_spec("pillow") is  None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pillow'])

if importlib.util.find_spec("dvc") is  None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'dvc'])


def get_latest_mlflow_model_version(model_name="lung_disease_prediction_model_finetuned", mlflow_host="http://mlflow:8080"):
    try:
        url = f"{mlflow_host}/api/2.0/mlflow/registered-models/get-latest-versions"
        response = requests.get(url, params={"name": model_name})
        response.raise_for_status()
        versions = response.json().get("model_versions", [])
        latest_version = max(int(v["version"]) for v in versions) if versions else 0
        return latest_version
    except Exception as e:
        print(f"Could not fetch latest model version from MLflow: {e}")
        return None

def move_to_dataset(db_path="/opt/airflow/uploads/predictions.db", datasets_root="/opt/airflow/datasets/chestxray/Data/train"):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Fetch all predictions
        cursor.execute("SELECT id, image_data, predicted_label, corrected_label FROM prediction ORDER BY id ASC;")
        rows = cursor.fetchall()

        if not rows:
            print("No prediction records found.")
            return

        for row in rows:
            _, image_data, predicted_label, corrected_label = row
            label = corrected_label if corrected_label else predicted_label
            label = label.upper()
            class_dir = os.path.join(datasets_root, label)
            os.makedirs(class_dir, exist_ok=True)
            print(class_dir)

            # Count existing images in the class folder
            existing_images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_index = len(existing_images)
            image_filename = f"{label}({image_index}).jpg"
            image_path = os.path.join(class_dir, image_filename)

            try:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image.save(image_path)
            except Exception as e:
                print(f"Failed to save image to {image_path}: {e}")
                continue

            # Delete the record after saving the imagedoc
            # cursor.execute("DELETE FROM prediction WHERE id = ?", (row[0],))

        subprocess.run(['dvc', 'add', 'datasets/'], check=True)
        subprocess.run(['git', 'add', 'datasets.dvc', '.gitignore'], check=True)
        subprocess.run(['git', 'commit', '-m', "New version of datasets"], check=True)

        latest_version = get_latest_mlflow_model_version()
        print(f"Latest MLflow model version: {latest_version}")
        if latest_version:
            subprocess.run(['git', 'tag', '-a', f'v{latest_version}', '-m', f'Version {latest_version}'], check=True)
        else:
            print("Skipping git tag: unable to determine MLflow model version.")

        conn.commit()
        print(f"Moved {len(rows)} images to dataset folders.")

    except Exception as e:
        print(f"Error while moving predictions to dataset: {e}")

    finally:
        cursor.close()
        conn.close()
