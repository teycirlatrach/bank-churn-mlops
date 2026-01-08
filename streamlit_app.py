import streamlit as st
import requests

# --- CONFIGURATION ---
st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦")
st.title("🏦 Prédiction de Désabonnement Bancaire")

# --- TON LIEN AZURE (Vérifie qu'il est correct) ---
API_URL = "https://bank-churn-api.proudbay-79afa30e.francecentral.azurecontainerapps.io/predict"

# --- INTERFACE UTILISATEUR (Ce que voit l'humain) ---
col1, col2 = st.columns(2)
with col1:
    credit_score = st.number_input("Credit Score", 300, 850, 600)
    # On laisse l'utilisateur choisir le pays en texte
    geography_input = st.selectbox("Pays", ["France", "Germany", "Spain"])
    # Le genre semble ne pas être utilisé par ton main.py, mais on le laisse pour la forme (ou usage futur)
    gender_input = st.selectbox("Genre", ["Male", "Female"])
    age = st.number_input("Age", 18, 100, 40)
    tenure = st.slider("Années (Tenure)", 0, 10, 3)

with col2:
    balance = st.number_input("Solde (Balance)", 0.0, 250000.0, 60000.0)
    num_of_products = st.slider("Produits", 1, 4, 2)
    has_cr_card = st.selectbox("Carte Crédit ?", [1, 0], format_func=lambda x: "Oui" if x==1 else "Non")
    is_active_member = st.selectbox("Membre Actif ?", [1, 0], format_func=lambda x: "Oui" if x==1 else "Non")
    estimated_salary = st.number_input("Salaire", 0.0, 200000.0, 50000.0)

# --- LOGIQUE DE TRADUCTION (Ce que l'API attend) ---
# On convertit le choix "Pays" en variables mathématiques (0 ou 1)
geo_germany = 1 if geography_input == "Germany" else 0
geo_spain = 1 if geography_input == "Spain" else 0

# --- BOUTON DE PRÉDICTION ---
if st.button("🔮 Prédire"):
    # On construit le JSON exactement comme ton main.py l'attend (regarde app.post("/predict"))
    payload = {
        "CreditScore": int(credit_score),
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_of_products),
        "HasCrCard": int(has_cr_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": float(estimated_salary),
        "Geography_Germany": int(geo_germany),  # Traduit
        "Geography_Spain": int(geo_spain)       # Traduit
        # Note: On n'envoie PAS "Geography" (texte) ni "Gender" car ton main.py ne semble pas les utiliser dans input_data
    }
    
    with st.spinner("Analyse en cours..."):
        try:
            # On affiche ce qu'on envoie pour débugger (optionnel)
            # st.write("Données envoyées :", payload) 
            
            resp = requests.post(API_URL, json=payload)
            
            if resp.status_code == 200:
                res = resp.json()
                churn_prob = res.get("churn_probability", 0)
                risk_level = res.get("risk_level", "Unknown")
                
                # Affichage joli du résultat
                st.write("---")
                if res.get("prediction") == 1:
                    st.error(f"🚨 **Risque ÉLEVÉ** de départ (Probabilité: {churn_prob:.1%})")
                    st.metric(label="Niveau de Risque", value=risk_level, delta="-Danger")
                else:
                    st.success(f"✅ **Client Fidèle** (Probabilité de départ: {churn_prob:.1%})")
                    st.metric(label="Niveau de Risque", value=risk_level, delta="Sûr")
                
                # Détails JSON
                with st.expander("Voir la réponse technique"):
                    st.json(res)
            else:
                st.error(f"Erreur API ({resp.status_code})")
                st.write("Message du serveur :", resp.text)
                
        except Exception as e:
            st.error(f"Erreur de connexion : {e}")