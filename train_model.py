import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def main():
    print(f"Démarrage de l'entraînement avec Python {sys.version.split()[0]}")

    # 1. Gestion des arguments (Crucial pour Azure ML Jobs)
    parser = argparse.ArgumentParser()
    # Par défaut, on cherche dans le dossier 'data' local
    parser.add_argument("--data", type=str, default="data", help="Dossier contenant le fichier bank_churn.csv")
    parser.add_argument("--n_estimators", type=int, default=100, help="Nombre d'arbres pour le RandomForest")
    parser.add_argument("--max_depth", type=int, default=15, help="Profondeur max des arbres")
    
    args = parser.parse_args()

    # 2. Activation du Tracking MLflow (Autologging)
    # Cela capture auto : métriques, paramètres et le modèle
    mlflow.autolog()

    # 3. Chargement des données
    csv_path = os.path.join(args.data, "bank_churn.csv")
    print(f"Lecture des données depuis : {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        sys.exit(f"Erreur : Le fichier {csv_path} est introuvable. Avez-vous lancé generate_data.py ?")

    # 4. Préparation des données (Cleaning)
    # On enlève les colonnes inutiles pour la prédiction
    drop_cols = ['CustomerId', 'Surname']
    # Vérification si les colonnes existent avant de drop
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Séparation Features (X) / Target (y)
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Identification des colonnes
    categorical_features = ['Geography', 'Gender']
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Features numériques : {numerical_features}")
    print(f"Features catégorielles : {categorical_features}")

    # 5. Création du Pipeline de prétraitement
    # C'est la méthode moderne : on encapsule le scaling et l'encodage avec le modèle
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Pipeline complet : Preprocessing + Modèle
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        ))
    ])

    # 6. Split Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Entraînement
    print("Début de l'entraînement du modèle...")
    with mlflow.start_run():
        clf.fit(X_train, y_train)
        
        # Prédictions
        y_pred = clf.predict(X_test)
        
        # Métriques manuelles (en plus de l'autolog)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy : {acc:.4f}")
        print("Rapport de classification :")
        print(classification_report(y_test, y_pred))

        # Enregistrement manuel du modèle dans le format 'mlflow' (optionnel avec autolog, mais recommandé)
        # Cela permet de charger le modèle plus tard avec mlflow.pyfunc.load_model()
        mlflow.sklearn.log_model(
            sk_model=clf, 
            artifact_path="bank_churn_model",
            registered_model_name="bank-churn-rf" # Enregistre directement dans le Model Registry si configuré
        )

    print("Entraînement terminé avec succès.")

if __name__ == "__main__":
    main()