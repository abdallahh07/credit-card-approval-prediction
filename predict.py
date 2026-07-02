import joblib
import pandas as pd
from pathlib import Path
from credit_model.config.config import config
from credit_model.processing.features import AgeTransformer, EmploymentTransformer

def load_pipeline():
    save_path = Path(__file__).parent / config.pipeline_save_file
    return joblib.load(save_path)

def make_prediction(input_data: dict) -> dict:
    pipeline = load_pipeline()
    
    df = pd.DataFrame([input_data])
    
    df = AgeTransformer().fit_transform(df)
    df = EmploymentTransformer().fit_transform(df)
    
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]
    
    return {
        "prediction": "Bad Client" if prediction == 1 else "Good Client",
        "probability": round(float(probability), 4)
    }