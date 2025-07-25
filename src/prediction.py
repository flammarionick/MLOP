from src.preprocessing import preprocess_image
from src.model import load_model
import numpy as np

def predict(image_file):
    model = load_model()
    image = preprocess_image(image_file)
    image = image.reshape(1, -1)  # Flatten for classic ML models
    prediction = model.predict(image)
    return prediction[0]
