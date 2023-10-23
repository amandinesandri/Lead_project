import mlflow 
import uvicorn
import json
import pandas as pd 
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile
import urllib.request
import joblib
from consumer.py import consume_kafka_message

description = """
Real Time payment API helps you learn more about fraud. 
The goal of this api is to serve data that help managers to detect fraud


## Machine-Learning 

Where you can:
* `/predict` if one person is likely to quit the company
* `/batch-predict` where you can upload a file to get predictions for several employees


Check out documentation for more information on each endpoint. 
"""

tags_metadata = [
 
    {
        "name": "Predictions",
        "description": "Endpoints that uses our Machine Learning model for detecting attrition"
    }
]

app = FastAPI(
    title="👨‍💼 Payment API",
    description=description,
    version="0.1",
    contact={
        "name": "LEAD 23",
        "url": "https://jedha.co",
    },
    openapi_tags=tags_metadata
)




@app.post("/predict", tags=["Machine-Learning"])
async def predict():
    """
    Prediction for one observation. Endpoint will return a dictionnary like this:

    ```
    {'prediction': PREDICTION_VALUE[0,1]}
    ```

    You need to give this endpoint all columns values as dictionnary, or form data.
    """
payments = consume_kafka_message() #Supprimer la clé "IS FRAUD"


    # Read data 
    df = pd.DataFrame([payments])

    # Log model from mlflow 
    urllib.request.urlretrieve("http://supertibo.com/model/model.joblib", "model.joblib")
    # Load model as a PyFuncModel.
    model = joblib.load("model.joblib")
    
    prediction = model.predict(df)

    # Format response
    response = {"prediction": prediction.tolist()[0]}
    return response

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
