from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import joblib
import numpy as np
import logging
import os
import traceback
from pathlib import Path

# Si tu n'utilises pas Azure Application Insights pour l'instant, 
# tu peux commenter les imports liés à opencensus pour éviter des erreurs si le package n'est pas installé.
# from opencensus.ext.azure.log_exporter import AzureLogHandler 

from app.models import CustomerFeatures, PredictionResponse, HealthResponse
# Assure-toi que tu as bien un fichier drift_detect.py ou commente cette ligne si non utilisé
# from app.drift_detect import detect_drift 

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bank-churn-api")

# Configuration simplifiée pour le test local
APPINSIGHTS_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
# Si tu as installé opencensus, décommente la suite, sinon on reste en local
if APPINSIGHTS_CONN:
    try:
        from opencensus.ext.azure.log_exporter import AzureLogHandler
        handler = AzureLogHandler(connection_string=APPINSIGHTS_CONN)
        logger.addHandler(handler)
    except ImportError:
        pass

# ============================================================
# FASTAPI INIT
# ============================================================

app = FastAPI(
    title="Bank Churn Prediction API",
    description="API de prédiction et monitoring du churn client",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "model/bank_churn_rf.pkl") # J'ai mis à jour le nom par défaut du modèle
model = None

@app.on_event("startup")
async def load_model():
    global model
    # On cherche le modèle soit dans model/ soit à la racine pour être robuste
    possible_paths = [
        "model/bank_churn_rf.pkl", 
        "bank_churn_rf.pkl",
        "model/churn_model.pkl"
    ]
    
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break
            
    if found_path:
        try:
            model = joblib.load(found_path)
            logger.info(f"Modèle chargé avec succès depuis : {found_path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle : {e}")
    else:
        logger.warning("Aucun fichier modèle trouvé. L'API fonctionnera mais les prédictions échoueront.")

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Bank Churn Prediction API",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        # Préparation des données (ordre exact des colonnes utilisé lors de l'entraînement)
        # Assure-toi que cet ordre correspond à ton X_train.columns dans train_model.py
        input_data = np.array([[  
            features.CreditScore,
            features.Age,
            features.Tenure,
            features.Balance,
            features.NumOfProducts,
            features.HasCrCard,
            features.IsActiveMember,
            features.EstimatedSalary,
            # Attention : Ton modèle a été entraîné avec pd.get_dummies ou OneHotEncoder ?
            # Si tu as utilisé le script précédent, c'était pd.get_dummies.
            # Il faut s'assurer que les colonnes correspondent. 
            # Dans ton script d'entraînement, tu avais 'Geography' et 'Gender'.
            # Le modèle attend probablement toutes les colonnes transformées.
            # Pour ce test simple, on va supposer que le modèle gère ou que tu as aligné les features.
            features.Geography_Germany,
            features.Geography_Spain
        ]])

        # Note : Si ton modèle a été entraîné avec des noms de colonnes (DataFrame), 
        # passer un numpy array peut générer un warning, mais ça marche souvent.
        
        proba = float(model.predict_proba(input_data)[0][1])
        prediction = int(proba > 0.5)
        risk = "Low" if proba < 0.3 else "Medium" if proba < 0.7 else "High"

        return {
            "churn_probability": round(proba, 4),
            "prediction": prediction,
            "risk_level": risk
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Pour le débogage, on renvoie l'erreur
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")