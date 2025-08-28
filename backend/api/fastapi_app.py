from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
#from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from datetime import datetime
# from threat_analysis.pipeline import ThreatAnalysisPipeline
from backend.threat_analysis.pipeline import ThreatAnalysisPipeline
from backend.config import Config
import logging
import pandas as pd
import json
from pathlib import Path
import uvicorn
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import re
from pydantic import BaseModel, Field, field_validator

import sys
import os

from fastapi.security import APIKeyHeader
from fastapi import Depends, HTTPException, status
import secrets


# API Key Security
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def validate_api_key(api_key: str = Depends(api_key_header)):
    if not secrets.compare_digest(api_key, Config().API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

#from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from fastapi import FastAPI
from backend.api.routes import dashboard

app = FastAPI()

# Add CORS if frontend is on a different port
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add this line
app.include_router(dashboard.router)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize application
app = FastAPI()
pipeline = ThreatAnalysisPipeline(Config())

# Store recent predictions in memory (last 50)
recent_predictions: List[Dict[str, Any]] = []

# Define input schema
class NetworkFeature(BaseModel):
    features: List[float]

    @field_validator('features')
    def check_length(cls, v):
        if len(v) < 10:
            raise ValueError("Too few features sent to model")
        return v

# Define endpoint to predict threat
@app.post("/predict")
def predict_threat(data: NetworkFeature):
    prediction = pipeline.predict(data.features)

    # Construct full threat result
    threat_info = {
        "prediction": "Threat" if prediction == 1 else "Normal",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "raw_score": int(prediction)
    }

    # Keep only recent 50
    recent_predictions.append(threat_info)
    if len(recent_predictions) > 50:
        recent_predictions.pop(0)

    return threat_info

# ✅ REAL: Get recent predictions from memory
@app.get("/threats/recent")
def get_recent_threats():
    return {"threats": recent_predictions[::-1]} 




# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Security middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config().ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=600
)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize pipeline
pipeline = ThreatAnalysisPipeline(Config())

class NetworkFeature(BaseModel):
    """Individual network feature specification"""
    feature_name: str = Field(..., example="duration")
    value: float = Field(..., example=0.5)
    description: Optional[str] = Field(None, example="Connection duration in seconds")

class NetworkData(BaseModel):
    """Network traffic analysis request model"""
    timestamp: datetime = Field(..., example="2023-01-01T00:00:00Z")
    #source_ip: str = Field(..., regex=r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    source_ip: str = Field(..., pattern=r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    destination_ip: str = Field(..., pattern=r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    features: list[float]

    #destination_ip: str = Field(..., regex=r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    #features: List[NetworkFeature]
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('features')
    @classmethod
    def check_features(cls, v):
        if len(v) != 78:
            raise ValueError("Expected 78 features")
        return v

class MalwareSample(BaseModel):
    """Malware analysis request model"""
    sample_id: str = Field(..., min_length=32, max_length=64)
    features: Dict[str, float]
    file_type: Optional[str] = Field(None, example="PE32")
    first_seen: Optional[datetime] = None

class CVEAnalysisRequest(BaseModel):
    """CVE vulnerability analysis request model"""
    cve_items: List[Dict[str, Any]]
    analysis_context: Optional[str] = Field(
        None,
        description="Optional context about where this CVE is being analyzed"
    )

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded"},
        headers={"Retry-After": str(exc.retry_after)}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

@app.post("/analyze/network",
          response_model=Dict[str, Any],
          summary="Analyze network traffic",
          response_description="Network threat analysis results")
@limiter.limit("10/minute")
async def analyze_network(request: Request, data: NetworkData):
    """
    Analyze network traffic for potential threats with detailed feature breakdown.
    
    - **timestamp**: When the traffic was observed
    - **source_ip**: Source IP address (IPv4)
    - **destination_ip**: Destination IP address (IPv4)  
    - **features**: List of network features with values
    - **metadata**: Optional additional context
    """
    try:
        # Convert to DataFrame format expected by pipeline
        feature_dict = {f.feature_name: f.value for f in data.features}
        feature_dict.update({
            "timestamp": data.timestamp,
            "source_ip": data.source_ip,
            "destination_ip": data.destination_ip
        })
        
        df = pd.DataFrame([feature_dict])
        result = pipeline.analyze_multi_modal_threat(network_data=df)
        
        # Log the analysis
        logger.info(f"Network analysis completed for {data.source_ip} -> {data.destination_ip}")
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "result": jsonable_encoder(result)
        }
    except Exception as e:
        logger.error(f"Network analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Network analysis failed: {str(e)}"
        )

@app.post("/analyze/malware",
          response_model=Dict[str, Any],
          summary="Analyze malware sample",
          response_description="Malware analysis results")
@limiter.limit("5/minute")
async def analyze_malware(request: Request, data: MalwareSample):
    """
    Analyze malware sample characteristics and behavior.
    
    - **sample_id**: Unique identifier for the sample
    - **features**: Dictionary of malware features (e.g., EMBER format)
    - **file_type**: Optional file type information
    - **first_seen**: When the sample was first observed
    """
    try:
        # Validate feature count
        if len(data.features) < 100:
            raise ValueError("At least 100 malware features required")
            
        df = pd.DataFrame([data.features])
        result = pipeline.analyze_multi_modal_threat(malware_data=df)
        
        # Add sample metadata to results
        result['sample_metadata'] = {
            "sample_id": data.sample_id,
            "file_type": data.file_type,
            "first_seen": data.first_seen.isoformat() if data.first_seen else None
        }
        
        logger.info(f"Malware analysis completed for sample {data.sample_id}")
        
        return {
            "status": "success", 
            "timestamp": datetime.now().isoformat(),
            "result": jsonable_encoder(result)
        }
    except Exception as e:
        logger.error(f"Malware analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Malware analysis failed: {str(e)}"
        )

@app.post("/analyze/vulnerability",
          response_model=Dict[str, Any],
          summary="Analyze CVE vulnerabilities",
          response_description="Vulnerability assessment results")
@limiter.limit("20/minute")
async def analyze_vulnerability(request: Request, data: CVEAnalysisRequest):
    """
    Analyze CVE vulnerabilities for potential exploitability and impact.
    
    - **cve_items**: List of CVE items in NVD JSON format
    - **analysis_context**: Optional context about where these CVEs are being analyzed
    """
    try:
        if not data.cve_items:
            raise ValueError("No CVE items provided")
            
        result = pipeline.analyze_multi_modal_threat(cve_data=data.cve_items)
        
        # Add context to results
        if data.analysis_context:
            result['analysis_context'] = data.analysis_context
            
        logger.info(f"Vulnerability analysis completed for {len(data.cve_items)} CVEs")
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "result": jsonable_encoder(result)
        }
    except Exception as e:
        logger.error(f"Vulnerability analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Vulnerability analysis failed: {str(e)}"
        )

@app.post("/analyze/multi_modal",
          response_model=Dict[str, Any],
          summary="Multi-modal threat analysis",
          response_description="Combined threat analysis results")
@limiter.limit("5/minute")
async def analyze_multi_modal(
    request: Request,
    network: Optional[NetworkData] = None,
    malware: Optional[MalwareSample] = None,
    vulnerability: Optional[CVEAnalysisRequest] = None
):
    """
    Comprehensive threat analysis combining multiple data modalities.
    
    Accepts any combination of:
    - Network traffic data
    - Malware characteristics  
    - CVE vulnerability data
    
    Returns unified threat assessment.
    """
    try:
        # Prepare inputs
        inputs = {}
        
        if network:
            feature_dict = {f.feature_name: f.value for f in network.features}
            feature_dict.update({
                "timestamp": network.timestamp,
                "source_ip": network.source_ip,
                "destination_ip": network.destination_ip
            })
            inputs['network_data'] = pd.DataFrame([feature_dict])
            
        if malware:
            if len(malware.features) < 100:
                raise ValueError("At least 100 malware features required")
            inputs['malware_data'] = pd.DataFrame([malware.features])
            
        if vulnerability:
            if not vulnerability.cve_items:
                raise ValueError("No CVE items provided")
            inputs['cve_data'] = vulnerability.cve_items
            
        if not inputs:
            raise ValueError("No input data provided")
            
        # Run analysis
        result = pipeline.analyze_multi_modal_threat(**inputs)
        
        # Add metadata
        result['input_modalities'] = list(inputs.keys())
        
        logger.info("Multi-modal analysis completed")
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "result": jsonable_encoder(result)
        }
    except Exception as e:
        logger.error(f"Multi-modal analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/health",
         response_model=Dict[str, str],
         summary="Service health check",
         response_description="Service status")
async def health_check():
    """
    Check API health status.
    
    Returns:
        Dictionary with service status information
    """
    return {
        "status": "healthy",
        "version": app.version,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info",
         response_model=Dict[str, Any],
         summary="Get model information",
         response_description="Model metadata")
async def get_model_info():
    """
    Get information about the loaded threat detection model.
    
    Returns:
        Dictionary containing model architecture and performance characteristics
    """
    try:
        return {
            "model_type": pipeline.model.__class__.__name__,
            "input_dimensions": getattr(pipeline.model, 'input_dims', None),
            "modality_support": ["network", "malware", "cve"],
            "last_updated": Config().MODEL_LAST_UPDATED
        }
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model information unavailable"
        )
##########################################################################################        
@app.get("/threats", summary="Get historical intrusion threats")
async def get_all_threats():
    try:
        df = pipeline.load_intrusion_data()  # Add this helper if not present
        return {"threats": df.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Failed to fetch threats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch threats")


@app.get("/malware", summary="Get malware analysis results")
async def get_malware_samples():
    try:
        df = pipeline.load_malware_data()  # Add this helper if not present
        return {"malware": df.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Failed to fetch malware data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch malware")


@app.get("/vulnerabilities", summary="Get vulnerability reports")
async def get_vulnerability_data():
    try:
        df = pipeline.load_vulnerability_data()  # Add this helper if not present
        return {"vulnerabilities": df.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Failed to fetch vulnerabilities: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch vulnerabilities")


@app.get("/model", summary="Basic model description")
async def get_model_metadata():
    return {
        "name": "Advanced Multi-Modal Threat Detection",
        "version": "1.0.0",
        "framework": "PyTorch",
        "modalities": ["Network", "Malware", "CVE"]
    }


@app.get("/shap", summary="Static SHAP feature importance values")
async def get_shap_values():
    return {
        "features": ["Packet Size", "Protocol", "Duration"],
        "importances": [0.42, 0.31, 0.27]
    }

@app.get("/model")
def get_model_alias():
    return get_model_info()

@app.get("/shap")
def get_shap_summary():
    # Optional SHAP visualizer or explanation
    return {"detail": "SHAP explanation summary endpoint coming soon"}

@app.get("/threats")
def get_threats_alias():
    return get_recent_threats()

@app.get("/malware")
def malware_summary():
    return {"status": "ok", "message": "malware endpoint placeholder"}

@app.get("/vulnerabilities")
def vuln_summary():
    return {"status": "ok", "message": "vulnerabilities endpoint placeholder"}



if __name__ == "__main__":
    config = Config()
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        ssl_keyfile=config.SSL_KEY_PATH if config.USE_SSL else None,
        ssl_certfile=config.SSL_CERT_PATH if config.USE_SSL else None
    )