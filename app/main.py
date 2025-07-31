from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.retrain import retrain_cnn_model

import os
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from zipfile import ZipFile
from pathlib import Path

# ✅ Setup app
app = FastAPI()

# ✅ Define BASE_DIR for all path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Setup CORS for browser UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount /outputs if exists
outputs_dir = os.path.join(BASE_DIR, "outputs")
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

# ✅ Mount /static for frontend HTML
static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ✅ Serve the frontend page
@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

# ✅ Load your trained model
model_path = os.path.join(BASE_DIR, "models", "best_model.h5")
model = tf.keras.models.load_model(model_path)

# ✅ Set class labels
class_labels = ["Corn 1", "Apple 10", "Pineapple Mini", "Cherry 1", "Cucumber 1", "Tomato 9", "Apple Red 1"]

# ✅ Preprocess function
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((64, 64))  # Adjust this if you used a different size
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# ✅ Prediction endpoint
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
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})

# ✅ Retrain endpoint
RETRAIN_DIR = os.path.join(BASE_DIR, "retrain_data")
os.makedirs(RETRAIN_DIR, exist_ok=True)

@app.post("/retrain/")
async def retrain_model(file: UploadFile = File(...)):
    try:
        zip_path = os.path.join(RETRAIN_DIR, file.filename)
        
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RETRAIN_DIR)

        retrain_cnn_model(RETRAIN_DIR)

        # 👇 Reload model after retraining
        global model
        model = tf.keras.models.load_model(model_path)

        return {"message": "Retraining completed and model updated."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Retraining failed: {str(e)}"})