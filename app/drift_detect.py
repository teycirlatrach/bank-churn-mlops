import matplotlib
matplotlib.use("Agg")  # Important pour Azure (pas d'écran)
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
import os
from datetime import datetime

def detect_drift(reference_file, production_file, threshold=0.05, output_dir="drift_reports"):
    # On s'assure que le dossier de rapport existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lecture des fichiers
    try:
        ref = pd.read_csv(reference_file)
        prod = pd.read_csv(production_file)
    except FileNotFoundError as e:
        return {"error": f"Fichier introuvable: {str(e)}", "drift_detected": False}

    results = {}
    
    # On compare chaque colonne numérique
    numeric_cols = ref.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col != "Exited" and col in prod.columns:
            # Test de Kolmogorov-Smirnov (KS test)
            stat, p = ks_2samp(ref[col].dropna(), prod[col].dropna())
            
            is_drift = bool(p < threshold)
            
            results[col] = {
                "p_value": float(p),
                "statistic": float(stat),
                "drift_detected": is_drift
            }

    # Sauvegarde locale du rapport (optionnel dans un conteneur, mais utile pour debug)
    report_path = f"{output_dir}/drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    return results