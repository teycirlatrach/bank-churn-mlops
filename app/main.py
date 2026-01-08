from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import logging
import os
import traceback
from opencensus.ext.azure.log_exporter import AzureLogHandler
from app.models import CustomerFeatures, PredictionResponse, HealthResponse
# Import du nouveau module drift
from app.drift_detect import detect_drift

# -------------------------------------------------
# Logging & Application Insights
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bank-churn-api")

APPINSIGHTS_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if APPINSIGHTS_CONN:
    try:
        handler = AzureLogHandler(connection_string=APPINSIGHTS_CONN)
        logger.addHandler(handler)
        logger.info("✅ Application Insights connecté")
    except Exception as e:
        logger.error(f"Erreur connexion App Insights: {e}")
else:
    logger.warning("⚠️ Application Insights non configuré")

# -------------------------------------------------
# Initialisation FastAPI
# -------------------------------------------------
app = FastAPI(title="Bank Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Chargement du modèle
# -------------------------------------------------
# On cherche le modèle à plusieurs endroits pour être robuste
possible_paths = ["model/bank_churn_rf.pkl", "bank_churn_rf.pkl", "model/churn_model.pkl"]
MODEL_PATH = next((p for p in possible_paths if os.path.exists(p)), "model/bank_churn_rf.pkl")
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Modèle chargé depuis {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle : {e}")

# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible")
    try:
        # Transformation des données pour le modèle
        X = np.array([[
            features.CreditScore, features.Age, features.Tenure, 
            features.Balance, features.NumOfProducts, features.HasCrCard, 
            features.IsActiveMember, features.EstimatedSalary, 
            features.Geography_Germany, features.Geography_Spain
        ]])
        
        proba = float(model.predict_proba(X)[0][1])
        prediction = int(proba > 0.5)
        risk = "Low" if proba < 0.3 else "Medium" if proba < 0.7 else "High"
        
        # Envoi du log personnalisé à Azure
        logger.info("Prediction request", extra={
            "custom_dimensions": {
                "event_type": "prediction",
                "probability": proba,
                "risk_level": risk,
                "country_germany": features.Geography_Germany
            }
        })
        
        return {"churn_probability": round(proba, 4), "prediction": prediction, "risk_level": risk}
    except Exception as e:
        logger.error(f"Erreur prediction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drift/check", tags=["Monitoring"])
def check_drift(threshold: float = 0.05):
    try:
        # Chemins vers les fichiers de données (doivent être dans l'image Docker)
        results = detect_drift(
            reference_file="data/bank_churn.csv",
            production_file="data/production_data.csv",
            threshold=threshold
        )
        
        if "error" in results:
             raise HTTPException(status_code=404, detail=results["error"])

        drifted_cols = [col for col, res in results.items() if res.get("drift_detected")]
        
        # Log du drift vers Azure
        logger.warning("Drift Analysis", extra={
            "custom_dimensions": {
                "event_type": "drift_check",
                "drift_detected": len(drifted_cols) > 0,
                "drifted_columns": str(drifted_cols)
            }
        })

        return {"status": "success", "drift_detected_in": drifted_cols, "details": results}
    except Exception as e:
        logger.error(f"Drift error: {e}")
        raise HTTPException(status_code=500, detail=str(e))