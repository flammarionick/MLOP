import streamlit as st
import requests
from PIL import Image

st.title("🧠 ML Image Classifier")

st.subheader("📸 Upload Image for Prediction")
image_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if image_file:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        files = {'file': image_file.getvalue()}
        res = requests.post("http://localhost:8000/predict/", files=files)
        st.success(f"Prediction: {res.json()['prediction']}")

st.subheader("🔄 Retrain Model")
if st.button("Trigger Retrain"):
    res = requests.post("http://localhost:8000/retrain/")
    st.info(res.json()['status'])
