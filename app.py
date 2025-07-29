import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Modelos de classificação
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Modelos de regressão
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Layout
st.set_page_config(page_title="Projet IA1", layout="wide")

st.title("🧠 Projet Final - IA1 (Été 2025)")
st.markdown("Application de classification et de régression automatique.")

# Variáveis globais
modeles_entrainés = {}

# Onglets
onglet = st.sidebar.radio("Navigation", ["📁 Téléchargement", "🤖 Machine Learning", "🔮 Prédiction"])

# === 1. Téléchargement de Fichier ===
if onglet == "📁 Téléchargement":
  st.header("Télécharger un fichier CSV")
  fichier = st.file_uploader("Téléversez un fichier CSV", type=["csv"])
  
  if fichier:
    with st.spinner("Chargement du fichier..."):
      df = pd.read_csv(fichier)
      st.session_state['data'] = df
      st.success("Fichier chargé avec succès!")
      st.dataframe(df.head())

# === 2. Machine Learning ===
elif onglet == "🤖 Machine Learning":
  st.header("Apprentissage Automatique")

  if 'data' not in st.session_state:
    st.warning("Veuillez d'abord téléverser un fichier CSV.")
    st.stop()

  df = st.session_state['data']
  st.subheader("Choix du type de problème")
  type_probleme = st.radio("Type de tâche", ["Classification", "Régression"])
  
  st.subheader("Choix de la variable cible (output)")
  cible = st.selectbox("Sélectionner la variable cible", df.columns)

  X = df.drop(columns=[cible])
  y = df[cible]

  st.subheader("Choix de la méthode de normalisation")
  normalisation = st.selectbox("Méthode", ["Aucune", "StandardScaler", "MinMaxScaler", "Normalizer"])

  # Prétraitement
  X = pd.get_dummies(X)

  if normalisation == "StandardScaler":
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
  elif normalisation == "MinMaxScaler":
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
  elif normalisation == "Normalizer":
    scaler = Normalizer()
    X = scaler.fit_transform(X)
  else:
    X = X.values

  # Split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  if type_probleme == "Classification":
    st.subheader("Résultats des modèles de classification")
    modeles = {
      "Random Forest": RandomForestClassifier(),
      "SVM": SVC(),
      "Logistic Regression": LogisticRegression(),
      "Decision Tree": DecisionTreeClassifier()
    }

    resultat = []
    for nom, modele in modeles.items():
      modele.fit(X_train, y_train)
      y_pred = modele.predict(X_test)
      modeles_entrainés[nom] = modele

      resultat.append({
        "Modèle": nom,
        "Précision": accuracy_score(y_test, y_pred),
        "Rappel": recall_score(y_test, y_pred, average='macro'),
        "F1-Score": f1_score(y_test, y_pred, average='macro'),
        "Précision (Precision)": precision_score(y_test, y_pred, average='macro')
      })
    st.dataframe(pd.DataFrame(resultat).set_index("Modèle"))

  else:
    st.subheader("Résultats des modèles de régression")
    modeles = {
      "Linear Regression": LinearRegression(),
      "Random Forest": RandomForestRegressor(),
      "SVR": SVR(),
      "Decision Tree": DecisionTreeRegressor()
    }

    resultat = []
    for nom, modele in modeles.items():
      modele.fit(X_train, y_train)
      y_pred = modele.predict(X_test)
      modeles_entrainés[nom] = modele

      resultat.append({
        "Modèle": nom,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R²": r2_score(y_test, y_pred)
      })
    st.dataframe(pd.DataFrame(resultat).set_index("Modèle"))

  st.success("Entraînement terminé.")

# === 3. Interface de Prédiction ===
elif onglet == "🔮 Prédiction":
  st.header("Faire une prédiction")

  if not modeles_entrainés:
    st.warning("Aucun modèle disponible. Veuillez entraîner un modèle d'abord.")
    st.stop()

  modele_nom = st.selectbox("Choisissez un modèle entraîné", list(modeles_entrainés.keys()))
  modele = modeles_entrainés[modele_nom]

  st.subheader("Entrez les données pour prédiction")

  if 'data' not in st.session_state:
    st.warning("Pas de données disponibles.")
    st.stop()

  df = st.session_state['data']
  colonnes = list(df.drop(columns=[df.columns[-1]]).columns)
  valeurs = []

  for col in colonnes:
    val = st.text_input(f"{col}")
    valeurs.append(float(val) if val else 0.0)

  if st.button("Prédire"):
    donnees = np.array(valeurs).reshape(1, -1)
    prediction = modele.predict(donnees)
    st.success(f"Résultat de la prédiction : {prediction[0]}")
