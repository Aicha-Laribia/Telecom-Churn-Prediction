#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Integer, Float, String
from tqdm.auto import tqdm
import psycopg2



# Load the dataset
df = pd.read_csv('/workspaces/Telecom-Churn-Prediction/data/teleco_churn.csv')




dtypes = {
    'customerID': 'string',
    'gender': 'string',
    'SeniorCitizen': 'int8',
    'Partner': 'string',
    'Dependents': 'string',
    'tenure': 'int16',
    'PhoneService': 'string',
    'MultipleLines': 'string',
    'InternetService': 'string',
    'OnlineSecurity': 'string',
    'OnlineBackup': 'string',
    'DeviceProtection': 'string',
    'TechSupport': 'string',
    'StreamingTV': 'string',
    'StreamingMovies': 'string',
    'Contract': 'string',
    'PaperlessBilling': 'string',
    'PaymentMethod': 'string',
    'MonthlyCharges': 'float32',
    'TotalCharges': 'float32',
    'Churn': 'string'
}



def run():
    engine = create_engine('postgresql://root:root@localhost:5432/telecom_churn')
    df.to_sql(
        "telecom_churn",
        engine,
        if_exists="replace",
        index=False,
        dtype={
            "customerID": String,
            "tenure": Integer,
            "MonthlyCharges": Float
        }
    )
    
if __name__ == "__main__":
    run()
