import numpy as np
from PIL import Image

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((64, 64))  # Adjust based on model input
    return np.array(image) / 255.0
