from pydantic import BaseModel, Field

# ==========================================
# Modèles de Données (Input)
# ==========================================

class CustomerFeatures(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850, description="Score de crédit (300-850)")
    Age: int = Field(..., ge=18, le=100, description="Âge du client")
    Tenure: int = Field(..., ge=0, le=10, description="Années en tant que client")
    Balance: float = Field(..., ge=0.0, description="Solde du compte")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Nombre de produits bancaires")
    HasCrCard: int = Field(..., ge=0, le=1, description="Possède une carte de crédit (0=Non, 1=Oui)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Membre actif (0=Non, 1=Oui)")
    EstimatedSalary: float = Field(..., ge=0.0, description="Salaire estimé")
    
    # Features One-Hot Encoded attendues par ton modèle
    Geography_Germany: int = Field(..., ge=0, le=1, description="Géographie : Allemagne (0/1)")
    Geography_Spain: int = Field(..., ge=0, le=1, description="Géographie : Espagne (0/1)")

    # Configuration pour la documentation API (Swagger UI)
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "CreditScore": 619,
                    "Age": 42,
                    "Tenure": 2,
                    "Balance": 0.0,
                    "NumOfProducts": 1,
                    "HasCrCard": 1,
                    "IsActiveMember": 1,
                    "EstimatedSalary": 101348.88,
                    "Geography_Germany": 0,
                    "Geography_Spain": 0
                }
            ]
        }
    }


# ==========================================
# Modèles de Réponse (Output)
# ==========================================

class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probabilité de départ (0.0 à 1.0)")
    prediction: int = Field(..., description="Classe prédite (0=Reste, 1=Part)")
    risk_level: str = Field(..., description="Niveau de risque (Low, Medium, High)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool