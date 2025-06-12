import os
import joblib

def load_model(model_name='random_forest_model.pkl'):
    model_path = os.path.join(os.path.dirname(__file__), model_name)
    return joblib.load(model_path)

def predict_with_model(model, features):
    # features deve ser um array/lista 2D: [[...]]
    return float(model.predict([features])[0])
