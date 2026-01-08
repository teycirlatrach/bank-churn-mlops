import pandas as pd
import numpy as np
import os
import sys

# Vérification optionnelle de l'environnement
print(f"Exécution avec Python version : {sys.version.split()[0]}")

# Configuration
N_ROWS = 10000  # Nombre de clients à générer
SEED = 42
OUTPUT_DIR = "data"
OUTPUT_FILE = "bank_churn.csv"

# Reproductibilité
np.random.seed(SEED)

def generate_bank_churn_data(n_rows):
    """
    Génère un DataFrame pandas simulant des données bancaires.
    Compatible pandas 2.x et Python 3.14+
    """
    print(f"Génération de {n_rows} lignes de données...")
    
    data = {
        'CustomerId': np.arange(15600000, 15600000 + n_rows),
        'Surname': [f"Client_{i}" for i in range(n_rows)],
        'CreditScore': np.random.randint(350, 850, n_rows),
        'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_rows),
        'Gender': np.random.choice(['Male', 'Female'], n_rows),
        'Age': np.random.randint(18, 92, n_rows),
        'Tenure': np.random.randint(0, 11, n_rows),
        'Balance': np.round(np.random.uniform(0, 250000, n_rows), 2),
        'NumOfProducts': np.random.choice([1, 2, 3, 4], n_rows, p=[0.5, 0.4, 0.05, 0.05]),
        'HasCrCard': np.random.choice([0, 1], n_rows),
        'IsActiveMember': np.random.choice([0, 1], n_rows),
        'EstimatedSalary': np.round(np.random.uniform(10000, 200000, n_rows), 2),
        # Target : Exited (20% de churn simulé)
        'Exited': np.random.choice([0, 1], n_rows, p=[0.8, 0.2])
    }

    df = pd.DataFrame(data)

    # Ajout d'un peu de réalisme : 30% des gens ont 0 sur leur compte
    mask_zero_balance = np.random.choice([True, False], n_rows, p=[0.3, 0.7])
    df.loc[mask_zero_balance, 'Balance'] = 0.0

    return df

if __name__ == "__main__":
    # 1. Création du dossier 'data' s'il n'existe pas (Sécurité)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Génération
    df_churn = generate_bank_churn_data(N_ROWS)
    
    # 3. Sauvegarde
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df_churn.to_csv(output_path, index=False)
    
    print(f"✅ Succès ! Fichier généré ici : {output_path}")
    print(f"Aperçu des données :\n{df_churn.head(3)}")