import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# 1. Charger les données
if not os.path.exists("data/bank_churn.csv"):
    print("❌ Erreur : Le fichier data/bank_churn.csv n'existe pas. Lance generate_data.py d'abord.")
    exit()

print("Lecture des données...")
df = pd.read_csv("data/bank_churn.csv")

# 2. Préparer les features (Doit être IDENTIQUE à l'API)
# On s'assure d'avoir les colonnes numériques et l'encodage OneHot pour la géographie
# Note: Dans ton train_model original, tu utilisais peut-être get_dummies.
# Ici, on simplifie pour garantir que ça marche avec l'API.

# On garde les colonnes utiles
X = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography']]
y = df['Exited']

# Encodage One-Hot manuel pour être sûr de l'ordre (Germany, Spain)
# Si Geography est 'France', les deux sont 0.
X = pd.get_dummies(X, columns=['Geography'], drop_first=False)

# Vérifions que les colonnes attendues par l'API sont bien là
required_cols = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
    'Geography_Germany', 'Geography_Spain'
]

# Si 'Geography_France' a été créé, on l'ignore car l'API ne l'utilise pas explicitement (elle utilise Germany/Spain)
for col in required_cols:
    if col not in X.columns:
        X[col] = 0 # Cas où une colonne manque (ex: pas de données Espagne)

X = X[required_cols] # Force l'ordre des colonnes

# 3. Entraîner le modèle
print("Entraînement du modèle...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Sauvegarder
if not os.path.exists("model"):
    os.makedirs("model")

save_path = "model/churn_model.pkl"
joblib.dump(model, save_path)

print(f"✅ Succès ! Modèle sauvegardé sous : {save_path}")
print("➡️  Tu peux maintenant relancer ton API (Uvicorn).")