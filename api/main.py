from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import UploadFile, File
from retrain import retrain_cnn_model
import shutil
import os
from zipfile import ZipFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

from fastapi.staticfiles import StaticFiles

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://127.0.0.1:5500"] if you serve HTML locally with Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


RETRAIN_DIR = "retrain_data"
os.makedirs(RETRAIN_DIR, exist_ok=True)


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("static/index.html")

# Load the model
model = tf.keras.models.load_model("models/best_model.h5")

# Class labels (update this with your real class names)
class_labels = ["Corn 1", "Apple 10", "Pineapple Mini", "Cherry 1", "Cucumber 1", "Tomato 9", "Apple Red 1"]

# Preprocess image to match model input
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((64, 64))  # ✅ MATCH YOUR TRAINING SIZE
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)
        
        predictions = model.predict(image_array)
        predicted_class = class_labels[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return JSONResponse({
            "prediction": predicted_class,
            "confidence": round(confidence, 3)
        })
    
    except Exception as e:
        # This ensures FastAPI still returns valid JSON on error
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )


@app.post("/retrain/")
async def retrain_model(file: UploadFile = File(...)):
    zip_path = f"{RETRAIN_DIR}/{file.filename}"
    
    # Save uploaded ZIP
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Unzip contents
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(RETRAIN_DIR)
    
    # Trigger retraining logic
    retrain_cnn_model(RETRAIN_DIR)

    return {"message": "Retraining completed and model updated."}