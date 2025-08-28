from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import pandas as pd
from backend.threat_analysis.intrusion_detector import detect_intrusion

router = APIRouter()

class IntrusionRequest(BaseModel):
    Attack: str
    count: int
    srv_diff_host_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_same_srv_rate: float
    same_srv_port: int
    flag: str
    last_flag: str

@router.post("/predict")
def predict_intrusion(data: IntrusionRequest):
    df = pd.DataFrame([data.dict()])
    prediction, probabilities = detect_intrusion(df)
    return {
        "attack_class": str(prediction[0]),
        "probability": float(max(probabilities[0]))
    }
