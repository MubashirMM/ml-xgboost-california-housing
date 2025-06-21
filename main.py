import os
from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Load model
model = joblib.load("xgb_model.pkl")

app = FastAPI()

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def read_root():
    return {"message": "Welcome to House Price Prediction API"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    data = [[
        features.MedInc, features.HouseAge, features.AveRooms,
        features.AveBedrms, features.Population,
        features.AveOccup, features.Latitude, features.Longitude
    ]]
    prediction = model.predict(data)
    return {"predicted_price": float(prediction[0])}
