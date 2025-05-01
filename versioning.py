import requests
from fastapi import FastAPI
import subprocess
import uvicorn


def get_latest_mlflow_model_version(model_name="lung_disease_prediction_model_finetuned", mlflow_host="http://localhost:8080"):
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


app = FastAPI()

@app.post("/perform_dvc_git_operations")
async def perform_dvc_git_operations():
    try:
        print("Performing DVC and Git operations...")

        subprocess.run(['dvc', 'add', 'datasets/'])
        subprocess.run(['git', 'add', 'datasets.dvc', '.gitignore'])
        subprocess.run(['git', 'commit', '-m', "New version of datasets"])

        latest_version = get_latest_mlflow_model_version()
        print(f"Latest MLflow model version: {latest_version}")
        if latest_version:
            subprocess.run(['git', 'tag', '-a', f'v{latest_version}', '-m', f'Version {latest_version}'], check=True)
            subprocess.run(['git', 'push', '--tags'], check=True)
        else:
            print("Skipping git tag: unable to determine MLflow model version.")


        return {"message": "DVC and Git operations completed successfully."}

    except subprocess.CalledProcessError as e:
        return {"error": f"An error occurred during DVC/Git operations: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9200)
