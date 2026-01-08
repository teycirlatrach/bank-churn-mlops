from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import logging
import os
import traceback
# Imports pour le cache et le hashing (Module 7)
from functools import lru_cache
import hashlib
import json

# Imports Azure Monitor (Module 6)
from opencensus.ext.azure.log_exporter import AzureLogHandler
from app.models import CustomerFeatures, PredictionResponse, HealthResponse
# Import Drift (Module 6)
# Assure-toi que le fichier app/drift_detect.py existe bien
try:
    from app.drift_detect import detect_drift
except ImportError:
    detect_drift = None

# -------------------------------------------------
# Logging & Configuration
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bank-churn-api")

APPINSIGHTS_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if APPINSIGHTS_CONN:
    try:
        handler = AzureLogHandler(connection_string=APPINSIGHTS_CONN)
        logger.addHandler(handler)
    except Exception as e:
        logger.warning(f"Impossible de charger Azure Log Handler: {e}")

# -------------------------------------------------
# FastAPI Init
# -------------------------------------------------
app = FastAPI(
    title="Bank Churn Prediction API",
    version="1.0.0",
    description="API optimisée avec Cache et Monitoring"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Chargement du Modèle
# -------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model/bank_churn_rf.pkl")
model = None

@app.on_event("startup")
async def load_model():
    global model
    possible_paths = ["model/bank_churn_rf.pkl", "bank_churn_rf.pkl", "model/churn_model.pkl"]
    found_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if found_path:
        try:
            model = joblib.load(found_path)
            logger.info(f"✅ Modèle chargé depuis {found_path}")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle : {e}")
    else:
        logger.warning("⚠️ Aucun modèle trouvé.")

# -------------------------------------------------
# Optimisation : Cache (Module 7)
# -------------------------------------------------
def hash_features(features_dict: dict) -> str:
    """Crée une signature unique pour les données d'entrée"""
    return hashlib.md5(
        json.dumps(features_dict, sort_keys=True).encode()
    ).hexdigest()

@lru_cache(maxsize=1000)
def predict_cached(features_hash: str, features_json: str):
    """Effectue la prédiction et garde le résultat en mémoire"""
    # On décode les données (lru_cache a besoin d'arguments hashables comme des strings)
    features_dict = json.loads(features_json)
    
    # Reconstruction précise de l'array numpy
    input_data = np.array([[
        features_dict["CreditScore"],
        features_dict["Age"],
        features_dict["Tenure"],
        features_dict["Balance"],
        features_dict["NumOfProducts"],
        features_dict["HasCrCard"],
        features_dict["IsActiveMember"],
        features_dict["EstimatedSalary"],
        features_dict["Geography_Germany"],
        features_dict["Geography_Spain"]
    ]])
    
    if model is None:
        raise ValueError("Model not loaded")

    # Prédiction
    proba = float(model.predict_proba(input_data)[0][1])
    prediction = int(proba > 0.5)
    
    risk = "Low" if proba < 0.3 else "Medium" if proba < 0.7 else "High"
    
    return {
        "churn_probability": round(proba, 4),
        "prediction": prediction,
        "risk_level": risk
    }

# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    
    try:
        # 1. Préparation pour le cache
        features_dict = features.dict()
        features_hash = hash_features(features_dict)
        features_json = json.dumps(features_dict)
        
        # 2. Appel intelligent (Cache ou Calcul)
        result = predict_cached(features_hash, features_json)
        
        # 3. Monitoring (On loggue même si ça vient du cache pour les stats)
        logger.info("Prediction served", extra={
            "custom_dimensions": {
                "event_type": "prediction",
                "probability": result["churn_probability"],
                "risk_level": result["risk_level"],
                "input_hash": features_hash[:8]
            }
        })
        
        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drift/check", tags=["Monitoring"])
def check_drift(threshold: float = 0.05):
    if detect_drift is None:
         raise HTTPException(status_code=501, detail="Module drift non disponible")
         
    try:
        results = detect_drift(
            reference_file="data/bank_churn.csv",
            production_file="data/production_data.csv",
            threshold=threshold
        )
        return {"status": "success", "details": results}
    except Exception as e:
        logger.error(f"Drift error: {e}")
        raise HTTPException(status_code=500, detail=str(e))