import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from training.model import resnet
import uvicorn
import time
from training.analysis import plot_tsne, plot_distribution
from io import BytesIO
from sqlmodel import SQLModel, Field, Session, select, create_engine
from typing import Optional
import datetime
import os
from prometheus_client import start_http_server, Summary
from prometheus_client import Counter, Gauge
import subprocess
import requests


from prometheus_client import disable_created_metrics
disable_created_metrics()



app = FastAPI()

# Database setup: define path and initialize directory
DATABASE_URL = "sqlite:///./uploads/predictions.db"
os.makedirs("uploads", exist_ok=True)
engine = create_engine(DATABASE_URL)

# Prediction table model for storing image metadata and prediction results
class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    image_data: bytes
    predicted_label: str
    corrected_label: Optional[str] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

# Create database tables if they don't exist
SQLModel.metadata.create_all(engine)

# Helper function to get a database session
def get_session():
    return Session(engine)

# Mount static directories for serving frontend assets and analysis outputs
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/analysis", StaticFiles(directory="analysis"), name="analysis")

# Jinja2 templates setup for HTML rendering
templates = Jinja2Templates(directory="templates")

# Prometheus counter to track how many times the prediction endpoint is called
api_call = Counter('api_call_counter', 'number of times the api was called to predict')

# Load model onto appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet(in_planes=3, outputs=3).to(device)
model.load_state_dict(torch.load("model/trained_model.pt", map_location=device))
model.eval()

# Class labels for prediction
classes = ["COVID-19", "NORMAL", "PNEUMONIA"]

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Number of images to retain in homepage history
IMAGE_HISTORY_LIMIT = 7

# Retrieve the most recent uploads and associated predictions from the database
def get_uploaded_images():
    with get_session() as session:
        images = session.exec(select(Prediction).order_by(Prediction.id.desc()).limit(IMAGE_HISTORY_LIMIT)).all()
        uploads = []
        for pred in images[::-1]:  # Reverse to display oldest first
            uploads.append({
                "image": f"/image/{pred.id}",
                "prediction": pred.predicted_label
            })
        return uploads

# Homepage route: renders recent predictions
@app.get("/")
def read_root(request: Request):
    images = get_uploaded_images()
    return templates.TemplateResponse("index.html", {"request": request, "images": images})

# API endpoint to get previous uploads in JSON format
@app.get("/get_previous_uploads/")
def get_previous_uploads():
    return {"previous_uploads": get_uploaded_images()}

# Upload image and return model prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    api_call.inc()
    image = Image.open(file.file).convert("RGB")
    image = image.resize((128, 128))

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = classes[output.argmax(dim=1).item()]

    # Convert image to byte stream
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    # Save prediction result in database
    with get_session() as session:
        pred = Prediction(filename=file.filename, image_data=image_bytes, predicted_label=predicted_class)
        session.add(pred)
        session.commit()
        session.refresh(pred)
        image_id = pred.id

    previous_uploads = get_uploaded_images()

    return {
        "image": f"/image/{image_id}",
        "prediction": predicted_class,
        "filename": f"uploaded_{image_id}.png",
        "previous_uploads": previous_uploads
    }

# Return image content from database based on image ID
@app.get("/image/{image_id}")
def get_image(image_id: int):
    with get_session() as session:
        pred = session.exec(select(Prediction).where(Prediction.id == image_id)).first()
        if pred:
            return StreamingResponse(BytesIO(pred.image_data), media_type="image/png")
        return JSONResponse(content={"error": "Image not found"}, status_code=404)

# Save user feedback/corrected label to the database
@app.post("/feedback/")
async def feedback(image_filename: str = Form(...), correct_prediction: str = Form(...)):
    try:
        image_id = int(image_filename.replace("uploaded_", "").replace(".png", ""))
    except ValueError:
        return {"message": "Invalid filename format."}

    with get_session() as session:
        pred = session.exec(select(Prediction).where(Prediction.id == image_id)).first()
        if pred:
            if pred.corrected_label is None:
                pred.corrected_label = correct_prediction
                session.add(pred)
                session.commit()
    return {"message": "Feedback saved."}


# Trigger fine-tuning of model using subprocess
@app.get("/finetune/")
async def finetune_model():
    subprocess.run(["python3", "tuning/finetune.py","--run_name", "tuning_run_1", "--num_epochs", "10", "--learning_rate", "0.0003"])
    return {"message": "Model retraining started."}
    
# Simulate retraining progress as an event stream for client-side progress bars
@app.get("/retrain/")
async def retrain_model():
    def event_stream():
        process = subprocess.Popen(
            ["python3", "training/train.py", "--run_name", "user_initiated_run", "--num_epochs", "10", "--learning_rate", "0.0003", "--batch_size", "32", "--eval_frequency", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output line by line
        for line in iter(process.stdout.readline, ''):
            if "Epoch [" in line:
                # Extract epoch number from line like "Epoch [1/10]"
                try:
                    prefix = line.strip().split("Epoch [")[1]
                    epoch_number = prefix.split("/")[0]
                    yield f"data: {epoch_number}\n\n"
                except Exception:
                    continue

        process.stdout.close()
        process.wait()

        yield "data: Training complete\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
# Render analysis page with interactive plots
@app.get("/experimental-analysis")
def analysis_page(request: Request):
    return templates.TemplateResponse("analysis.html", {"request": request})

# Run t-SNE dimensionality reduction and return results
@app.post("/run_tsne")
def run_tsne():
    try:
        results = plot_tsne()
        return JSONResponse(content={"status": "success", "data": results})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

# Run distribution plot generation and return results
@app.post("/run_distribution")
def run_distribution():
    try:
        results = plot_distribution()
        return JSONResponse(content={"status": "success", "data": results})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

# Start Prometheus metrics server and launch FastAPI app using Uvicorn
if __name__ == "__main__":
    start_http_server(9000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
