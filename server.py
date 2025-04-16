import os
from fastapi import FastAPI, File, UploadFile,Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import resnet
import uvicorn
import time
from analysis import plot_tsne,plot_distribution
from fastapi.responses import JSONResponse
import prometheus_client

app = FastAPI()

# Directories
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Static and template setup
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/analysis", StaticFiles(directory="analysis"), name="analysis")

templates = Jinja2Templates(directory="templates")

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
    images = sorted([f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".png")])[-IMAGE_HISTORY_LIMIT:]
    uploads = []

    for img_name in images:
        pred_file = img_name.replace(".png", ".txt")
        pred_path = os.path.join(UPLOAD_FOLDER, pred_file)

        prediction = "Unknown"
        if os.path.exists(pred_path):
            with open(pred_path, "r") as f:
                prediction = f.read().strip()

        uploads.append({
            "image": f"uploads/{img_name}",
            "prediction": prediction
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
    image_count = len([f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".png")]) + 1
    image_filename = f"uploaded_{image_count}.png"
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)

    # Save image
    image = Image.open(file.file).convert("RGB")
    image = image.resize((128, 128))
    image.save(image_path)

    # Predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = classes[output.argmax(dim=1).item()]

    # Save prediction
    pred_filename = image_filename.replace(".png", ".txt")
    pred_path = os.path.join(UPLOAD_FOLDER, pred_filename)
    with open(pred_path, "w") as f:
        f.write(predicted_class)

    previous_uploads = get_uploaded_images()

    return {
        "image": f"/uploads/{image_filename}",
        "prediction": predicted_class,
        "previous_uploads": previous_uploads
    }

@app.post("/feedback/")
async def feedback(correct_prediction: str = Form(...)):
    image_count = len([f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".png")]) + 1
    text_filename = f"uploaded_{image_count-1}.txt"
    pred_path = os.path.join(UPLOAD_FOLDER, text_filename)
    with open(pred_path, "a") as f:
        f.write(f"\n{correct_prediction}")
    return {"message": "Correction saved."}

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
        # Assume plot_tsne() returns a dict like:
        # { "train": "/analysis/tsne/tsne_train.png", "test": "/analysis/tsne/tsne_test.png" }
        results = plot_tsne()
        return JSONResponse(content={"status": "success", "data": results})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.post("/run_distribution")
def run_distribution():
    try:
        # Assume plot_distribution returns: {"train": "/analysis/distribution/dist_train.png", "test": "/analysis/distribution/dist_test.png"}
        results = plot_distribution()
        return JSONResponse(content={"status": "success", "data": results})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
