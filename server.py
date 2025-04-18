import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import resnet
import uvicorn
import time
from analysis import plot_tsne, plot_distribution
from io import BytesIO
from sqlmodel import SQLModel, Field, Session, select, create_engine
from typing import Optional
import datetime
import os
from prometheus_client import start_http_server, Summary
from prometheus_client import Counter, Gauge




from prometheus_client import disable_created_metrics
disable_created_metrics()



app = FastAPI()

# Database setup
DATABASE_URL = "sqlite:///./uploads/predictions.db"
os.makedirs("uploads",exist_ok=True)
engine = create_engine(DATABASE_URL)

class Prediction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    image_data: bytes
    predicted_label: str
    corrected_label: Optional[str] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

SQLModel.metadata.create_all(engine)

def get_session():
    return Session(engine)

# Static and template setup
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/analysis", StaticFiles(directory="analysis"), name="analysis")

templates = Jinja2Templates(directory="templates")



api_call = Counter('api_call_counter', 'number of times the api was called to predict')


# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet(in_planes=3, outputs=3).to(device)
model.load_state_dict(torch.load("model/trained_model.pt", map_location=device))
model.eval()

classes = ["COVID-19", "Normal", "Pneumonia"]

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

IMAGE_HISTORY_LIMIT = 7

def get_uploaded_images():
    with get_session() as session:
        images = session.exec(select(Prediction).order_by(Prediction.id.desc()).limit(IMAGE_HISTORY_LIMIT)).all()
        uploads = []
        for pred in images[::-1]:
            uploads.append({
                "image": f"/image/{pred.id}",
                "prediction": pred.predicted_label
            })
        return uploads

@app.get("/")
def read_root(request: Request):
    images = get_uploaded_images()
    return templates.TemplateResponse("index.html", {"request": request, "images": images})

@app.get("/get_previous_uploads/")
def get_previous_uploads():
    return {"previous_uploads": get_uploaded_images()}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    api_call.inc()
    image = Image.open(file.file).convert("RGB")
    image = image.resize((128, 128))

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = classes[output.argmax(dim=1).item()]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

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

@app.get("/image/{image_id}")
def get_image(image_id: int):
    with get_session() as session:
        pred = session.exec(select(Prediction).where(Prediction.id == image_id)).first()
        if pred:
            return StreamingResponse(BytesIO(pred.image_data), media_type="image/png")
        return JSONResponse(content={"error": "Image not found"}, status_code=404)

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

@app.get("/retrain/")
async def retrain_model():
    def event_stream():
        for i in range(0, 101, 5):
            yield f"data: {i}\n\n"
            time.sleep(0.5)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/experimental-analysis")
def analysis_page(request: Request):
    return templates.TemplateResponse("analysis.html", {"request": request})

@app.post("/run_tsne")
def run_tsne():
    try:
        results = plot_tsne()
        return JSONResponse(content={"status": "success", "data": results})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.post("/run_distribution")
def run_distribution():
    try:
        results = plot_distribution()
        return JSONResponse(content={"status": "success", "data": results})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    start_http_server(9000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
