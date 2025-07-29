# src/prediction.py
import tensorflow as tf
import numpy as np
import cv2

# Load model only once
model = tf.keras.models.load_model("models/best_model.h5")

# Class labels must match training
class_labels = ["Corn 1", "Apple 10", "Pineapple Mini", "Cherry 1", "Cucumber 1", "Tomato 9", "Apple Red 1"]

def preprocess_image(image_bytes):
    # Read and decode image
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Resize and normalize
    image = cv2.resize(image, (100, 100))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    return image

def predict_image(image_bytes):
    image = preprocess_image(image_bytes)
    prediction = model.predict(image)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    predicted_label = class_labels[predicted_index]
    return {
        "label": predicted_label,
        "confidence": round(confidence, 4)
    }
