
# IMPORTS

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib


# LOAD MODEL

model = joblib.load("ecommerce_model.pkl")


# INIT APP

app = FastAPI(
    title="Ecommerce Prediction API",
    description="Predict customer purchase behavior",
    version="1.0"
)


# INPUT SCHEMA

class EcommerceInput(BaseModel):
    age: float
    gender: str
    device_type: str
    time_on_site: float
    pages_viewed: float
    previous_purchases: float
    cart_items: float
    discount_seen: float
    ad_clicked: float
    returning_user: float
    avg_session_time: float
    bounce_rate: float


# ENCODING (IMPORTANT)

def encode_input(data):
    gender_map = {"Male": 0, "Female": 1}
    device_map = {"Mobile": 0, "Desktop": 1, "Tablet": 2}

    return [
        data.age,
        gender_map.get(data.gender, 0),
        device_map.get(data.device_type, 0),
        data.time_on_site,
        data.pages_viewed,
        data.previous_purchases,
        data.cart_items,
        data.discount_seen,
        data.ad_clicked,
        data.returning_user,
        data.avg_session_time,
        data.bounce_rate
    ]


# PREDICT ENDPOINT

@app.post("/predict")
def predict(data: EcommerceInput):

    input_array = np.array([encode_input(data)])

    prediction = model.predict(input_array)[0]

    return {
        "prediction": int(prediction),
        "result": "Will Purchase" if prediction == 1 else "Will Not Purchase"
    }