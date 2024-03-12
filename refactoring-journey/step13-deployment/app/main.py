import pickle
import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

script_path = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
default_model_path = os.path.join(script_path, "..", "results", "models", "regression", "full", "best_model.pickle")
model_path = os.environ.get("MODEL_PATH", default_model_path)

with open(model_path, 'rb') as f:
    model = pickle.load(f)


class PredictionInput(BaseModel):
    artist_name: str
    track_name: str
    track_id: str
    year: int
    genre: str
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: int
    time_signature: int


@app.get("/model_info/")
def model_info():
    return str(model)


@app.post("/predict/")
def predict(input_data: PredictionInput):
    data = pd.DataFrame([dict(input_data)])
    prediction = model.predict(data)
    return prediction.to_dict(orient="records")
