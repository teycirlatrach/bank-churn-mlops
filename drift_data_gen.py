import pandas as pd
import numpy as np
import os

def generate_drifted_data(
    reference_file="data/bank_churn.csv",
    output_file="data/production_data.csv",
    drift_level="medium"
):
    os.makedirs("data", exist_ok=True)
    # On essaie de lire le fichier, peu importe où on se trouve
    if not os.path.exists(reference_file):
        print(f"❌ Erreur : Impossible de trouver {reference_file}")
        return

    ref = pd.read_csv(reference_file)
    prod = ref.copy()

    # Simulation de drift (modification des moyennes)
    intensity = 0.15 if drift_level == "medium" else 0.3
    drift_features = ["CreditScore", "Age", "Balance", "EstimatedSalary"]

    for col in drift_features:
        if col in prod.columns:
            std = prod[col].std()
            prod[col] = prod[col] + np.random.normal(0, std * intensity, size=len(prod))

    prod.to_csv(output_file, index=False)
    print(f"✅ Fichier généré : {output_file}")

if __name__ == "__main__":
    generate_drifted_data()