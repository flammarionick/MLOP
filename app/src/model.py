import joblib

def load_model(path='models/model.pkl'):
    model = joblib.load(path)
    return model
