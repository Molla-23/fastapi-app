
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")

app = FastAPI()

class InputData(BaseModel):
    features: list
@app.get("/")
def read_root():
    return {"message": "FastAPI is running. Go to /predict to use the model."}
@app.post("/predict")
def predict(data: InputData):
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)
    return {"prediction": int(prediction[0])}
