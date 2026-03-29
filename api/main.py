from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent # Get the parent directory of the current file

with open(BASE_DIR / "Models" / "ada_boost_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

with open(BASE_DIR / "Models" / "Metadata.json", "r") as f:
    metadata = json.load(f)
    

FEATURE_COLUMNS = metadata["feature_columns"]
THRESHOLD       = metadata["threshold"]

app=FastAPI(
    title="telecom_churn_prediction_api",
    description="Predicts whether a customer will churn based on their profile and usage patterns.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="api/static"), name="static")

class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int                  # 0 or 1
    Partner: str                        # "Yes" or "No"
    Dependents: str                     # "Yes" or "No"
    MultipleLines: str                  # "Yes", "No", "No phone service"
    InternetService: str                # "DSL", "Fiber optic", "No"
    OnlineSecurity: str                 # "Yes", "No", "No internet service"
    OnlineBackup: str                   # "Yes", "No", "No internet service"
    DeviceProtection: str               # "Yes", "No", "No internet service"
    TechSupport: str                    # "Yes", "No", "No internet service"
    StreamingTV: str                    # "Yes", "No", "No internet service"
    StreamingMovies: str                # "Yes", "No", "No internet service"
    Contract: str                       # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str               # "Yes" or "No"
    PaymentMethod: str                # "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    
def preprocess(customer: CustomerData) -> pd.DataFrame:
    """
    Converts raw customer input into the exact format the pipeline expects.
    
    Why do we need this?
    The pipeline's scaler expects numeric columns only.
    But the user sends raw strings like "Yes"/"No" for categorical fields.
    We need to apply the same get_dummies encoding you did in the notebook.
    """
    # Convert input to a single-row DataFrame
    raw = pd.DataFrame([customer.model_dump()])

    # Apply same encoding as notebook
    # SeniorCitizen is already int, so no change needed there
    df_encoded = pd.get_dummies(raw, drop_first=True)

    # Align columns to match exactly what the model was trained on.
    # reindex adds missing columns as 0, drops any extra columns.
    # This is critical — the model expects columns in a specific order.
    df_encoded = df_encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # Convert booleans to int (same as notebook)
    for col in df_encoded.select_dtypes(include="bool"):
        df_encoded[col] = df_encoded[col].astype(int)

    return df_encoded

@app.get("/")
def root():
    return FileResponse("api/static/index.html")

@app.post("/predict")
def predict(customer: CustomerData):
   # Preprocess input
    X = preprocess(customer)

    # Get probability from pipeline (scaler + AdaBoost in one call)
    probability = pipeline.predict_proba(X)[0][1]

    # Apply threshold
    prediction = bool(probability >= THRESHOLD)

    # Add a human-readable risk level — useful for business users
    if probability >= 0.7:
        risk_level = "High"
    elif probability >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "churn_probability": round(float(probability), 4),
        "churn_prediction":  prediction,
        "threshold":         THRESHOLD,
        "risk_level":        risk_level
    }
    
@app.get("/model-info")
def model_info():
    """Returns information about the model currently loaded."""
    return {
        "model_type":    "AdaBoostClassifier",
        "threshold":     THRESHOLD,
        "n_features":    len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS
    }