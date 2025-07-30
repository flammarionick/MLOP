from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np
from PIL import Image

def load_training_data(data_folder='data/train'):
    X, y = [], []
    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)
        for file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, file)
            try:
                img = Image.open(img_path).resize((64, 64))
                X.append(np.array(img).flatten())
                y.append(label)
            except:
                continue
    return np.array(X), np.array(y)

def retrain_model():
    X, y = load_training_data()
    clf = RandomForestClassifier()
    clf.fit(X, y)
    joblib.dump(clf, 'models/model.pkl')
    return True
