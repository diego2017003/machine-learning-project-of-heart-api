from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pickle
import wandb
import pandas as pd
import sys

from source.mlops.api.preprocessing import *

app = FastAPI()

# Creating a class for the attributes input to the ML model.
class Heart_metrics(BaseModel):
    BMI: float
    Smoking: str
    AlcoholDrinking: str
    Stroke: str
    PhysicalHealth: float
    MentalHealth: float
    DiffWalking: str
    Sex: str
    AgeCategory: str
    Race: str
    Diabetic: str
    PhysicalActivity: str
    GenHealth: str
    SleepTime: float
    Asthma: str
    KidneyDisease: str
    SkinCancer: str

    class config:
        schema_extra = {
            "example": {
                "BMI": 16.4,
                "Smoking": "Yes",
                "AlcoholDrinking": "No",
                "Stroke": "No",
                "PhysicalHealth": 3.0,
                "MentalHealth": 30.0,
                "DiffWalking": "No",
                "Sex": "Female",
                "AgeCategory": "55-59",
                "Race": "White",
                "Diabetic": "Yes",
                "PhysicalActivity": "Yes",
                "GenHealth": "Very good",
                "SleepTime": 5.0,
                "Asthma": "Yes",
                "KidneyDisease": "No",
                "SkinCancer": "Yes",
            }
        }


run = wandb.init(project="project_heart", job_type="api", save_code=True)


@app.get("/")
async def home():
    return {"Hello": "World"}


@app.post("/prediction")
async def heart_prediciton(data: Heart_metrics):
    run = wandb.init(project="project_heart", job_type="api")
    artifact = run.use_artifact(
        "diego25rm/project_heart/model_export:v4", type="pipeline_artifact"
    ).file()
    # try:
    loaded_model = joblib.load(artifact)
    X = pd.DataFrame([data])
    prediction = loaded_model.predict(X)
    return {"Prediction": prediction}
    # except:
    #     return artifact
