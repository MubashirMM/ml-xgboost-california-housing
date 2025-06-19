from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model=joblib.load("xgb_housing_price_model.pkl")

class HouseData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    
app=FastAPI()
@app.get("/")
def read_root():
    return{"Message":"California Housing price Prediction API"}    
@app.post("/predict")
def predict_price(data: HouseData):
    input_data = np.array([[data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
                            data.Population, data.AveOccup, data.Latitude, data.Longitude]])
    prediction = model.predict(input_data)[0]
    return {"predicted_price": float(prediction)}
