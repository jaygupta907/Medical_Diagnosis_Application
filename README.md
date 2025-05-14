# 🩺 AI-Powered Chest X-ray Diagnostic System
An end-to-end AI-powered medical diagnostic tool designed for chest X-ray classification.
The system allows healthcare professionals to upload chest X-ray images, receive AI predictions, provide feedback, and iteratively improve the diagnostic model.
![Alt text](assets/home.png)


## 🚀 Features
- FastAPI Web Service
    - REST API for uploading images, getting predictions, and submitting corrections.
    - Interactive web UI for user interaction and feedback.
- Image Storage & Label Tracking
    - Stores images and predictions in a SQLite database (uploads/predictions.db).
    - Tracks both initial predictions and user-corrected labels.
- Interactive Feedback Loop
    - Users can validate or correct AI predictions.
    - Only validated/corrected data is used for retraining.
- Retraining Pipeline
    - contains script to  retrain the AI model using validated data from the database.
    - Supports incremental improvement using human-in-the-loop feedback.
- Monitoring & Pipeline Management
    - Integrated with Prometheus for GPU and system monitoring.
    - Uses Airflow for pipeline orchestration and scheduled retraining (via Docker Compose).

## ⚙️ Instructions 

### Download model weights
Download the model weights from this link and stored them inside the model folder in root directory.

https://drive.google.com/drive/folders/1hkSzoSRtmHmOcgOWyV6NuBOMIMAYrypk?usp=sharing

### Using docker compose
```bash
docker compose up
```


### Using virtual environment
- Create Conda environment
```bash
conda env create -n medical python=3.10
conda activate medical
```
- install required libraries
```bash
pip install .
```

- setup mflow tracking
```bash
export MLFLOW_TRACKING_URI=http://localhost:2000
mlflow server --host 0.0.0.0 --port 2000
```
- Download the dataset
```bash
python training/download.py
```

- Train the model
```bash
python training/train.py
```

- Open the UI
```bash
python server.py
```

While using docker compose if you want to tag git pushes then run this file in the conda environment with docker compose up.
```bash
python versioning.py
```

## 📦 Project Structure: 
``` bash
├── airflow
|   ├── dags              # Airflow DAGs automatic finetuning 
├── analysis
|   ├── distribution      # Contains details about data distribution
|   ├── tsne              # Contains details about tnse encoding of the dataset
├── assets                # Static assets (images like home.png)
├── build                 # Build artifacts
├── datasets              # Dataset management utilities
├── model                 # Pre-trained or exported models
├── static                # Static files for web UI
├── templates             # HTML templates for FastAPI
├── training              # Custom training, dataset, and evaluation modules
├── tuning                # Fine-tuning and ML experiment utilities (MLflow integration)
├── uploads               # Database (`predictions.db`) and optionally raw uploaded images
├── Dockerfile            # Dockerfile for the FastAPI app
├── Dockerfile.airflow    # Dockerfile for Airflow service
├── docker-compose.yml    # Docker Compose file to run the entire stack
├── prometheus.yml        # Prometheus configuration
├── README.md
├── requirements.txt
├── server.py             # Runs fastAPI Server
├── setup.py              # setup file to install required packages
└── versioning.py         # Run this file after docker compose to tag git pushes
```