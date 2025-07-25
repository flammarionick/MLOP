from fastapi import FastAPI, UploadFile, File
from src.prediction import predict
from src.retrain.py import retrain_model

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ML Prediction API"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    prediction = predict(file.file)
    return {"prediction": prediction}

@app.post("/retrain/")
async def trigger_retrain():
    success = retrain_model()
    return {"status": "Retrained Successfully" if success else "Retraining Failed"}
