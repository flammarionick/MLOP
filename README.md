# Fruit Classifier 
A computer vision web application that classifies fruits using a Convolutional Neural Network (CNN). The app allows users to upload images of fruits for prediction and supports retraining the model using bulk image uploads via a user interface.


## Live Demo

**Deployed App:** [https://fruit-classifier-6dc2.onrender.com]

** Video Demo:** [YouTube Link]


## GitHub Repo

[https://github.com/flammarionick/MLOP]

## Project Description

This project demonstrates an end-to-end MLOps pipeline for an image classification task using FastAPI. It includes:
- A trained CNN model for fruit classification
- Web UI for user predictions and model retraining
- Automatic visualization updates
- Monitoring with logs
- Cloud deployment via Render



## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/flammarionick/MLOP.git

cd MLOP


### 2. Create and Activate a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install Requirements
pip install -r requirements.txt


### 4. Run the App Locally
uvicorn app.main:app --reload

### 5. Access in Browser
Navigate to: `http://127.0.0.1:8000`

##  Model File

- best_model.h5 - Trained CNN model saved in HDF5 format



## Visualizations

- Training Accuracy & Loss Curves  
- Class Distribution  
- Confidence Histogram  

These are automatically updated when retraining is completed.



##  Features

### Image Prediction
- Upload a single image and get the predicted fruit class

### Model Retraining
- Upload a zip file of new labeled images
- Preprocessing is triggered automatically
- CNN model is retrained and saved

### Real-time Monitoring
- Progress bar during retraining
- UI refresh for new visualizations


## Jupyter Notebook

- fruit_classifier_notebook.ipynb contains:
  - Data loading and preprocessing steps
  - CNN model architecture and training
  - Evaluation metrics (Accuracy, Loss, F1-Score, Precision, Recall)



## Load Test Results

- Tool used: **Locust**
- Request types tested: `POST /predict`, `POST /retrain`
- Observed stability under concurrent requests up to 20 users
- Average response time: ~350ms on `predict`, ~1.2s on retrain trigger
