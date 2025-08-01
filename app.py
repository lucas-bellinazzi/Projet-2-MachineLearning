import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

st.set_page_config(page_title="Projet IA1", layout="wide")

if 'data' not in st.session_state:
  st.session_state.data = None
if 'modeles_entraines' not in st.session_state:
  st.session_state.modeles_entraines = {}
if 'scaler' not in st.session_state:
  st.session_state.scaler = None
if 'feature_names' not in st.session_state:
  st.session_state.feature_names = []
if 'target' not in st.session_state:
  st.session_state.target = ''

st.sidebar.title("Navigation")
menu = st.sidebar.radio("", ["Téléchargement", "Apprendisage Automatique", "Prédiction"], label_visibility="collapsed")

# === 1. Téléchargement de Fichier ===
if menu == "Téléchargement":
  st.header("Téléchargement de Fichier")
  
  uploaded_file = st.file_uploader("Téléversez votre fichier CSV", type=["csv"])
  
  if uploaded_file is not None:
    try:
      if "pima" in uploaded_file.name.lower():
        col=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(uploaded_file, names=col)
      elif "housing" in uploaded_file.name.lower():
        col = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAXE','PTRATIO','B','LSTAT','MEDV']
        df = pd.read_csv(uploaded_file, names=col, delim_whitespace=True)
      else:
        df = pd.read_csv(uploaded_file)
      
      if df.empty:
        st.error("Erreur: Le fichier semble vide.")
      else:
        st.session_state.data = df
        st.success(f"Fichier chargé avec succès! Shape: {df.shape}")
        
        st.markdown("## Aperçu des données")
        st.dataframe(df)
        
        st.markdown("## Statistiques descriptives")
        st.write(df.describe())

        st.markdown("## Histogrammes")
        try:
          n_features = len(df.columns)
          n_rows = (n_features + 3) // 4
          
          fig, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(15, 3*n_rows))
          
          if n_rows == 1:
            axes = axes.reshape(1, -1)
          
          for i, column in enumerate(df.columns):
            row = i // 4
            col = i % 4
            df[column].hist(bins=15, ax=axes[row, col], grid=True)
            axes[row, col].set_title(column)
          
          for i in range(n_features, n_rows*4):
            row = i // 4
            col = i % 4
            fig.delaxes(axes[row, col])
          
          plt.suptitle("Histogrammes des caractéristiques", y=1.02)
          plt.tight_layout()
          st.pyplot(fig)
          
        except Exception as e:
          st.error(f"Erreur lors de la génération des histogrammes: {str(e)}")
        
        st.session_state.modeles_entraines = {}
        st.session_state.scaler = None
        st.session_state.feature_names = df.columns.tolist()
        
    except Exception as e:
      st.error(f"Erreur de lecture: {str(e)}")
      st.info("Conseil: Vérifiez que le fichier est un CSV valide avec des virgules comme séparateur.")

# === 2. Apprendisage Automatique ===
elif menu == "Apprendisage Automatique":
  st.header("Apprendisage Automatique")
  
  if st.session_state.data is None:
    st.warning("Veuillez d'abord téléverser un fichier CSV.")
    st.stop()
  
  df = st.session_state.data
  
  if df.empty:
    st.error("Erreur: Les données sont vides.")
    st.stop()
  
  col1, col2 = st.columns(2)
  with col1:
    problem_type = st.radio("Type de problème", ["Classification", "Régression"])
  with col2:
    target_var = st.selectbox("Variable cible", df.columns)
  
  st.session_state.target = target_var
  
  X = df.drop(columns=[target_var])
  y = df[target_var]
  
  for col in X.select_dtypes(include=['object']).columns:
    try:
      X[col] = pd.to_numeric(X[col], errors='raise')
    except:
      X = pd.get_dummies(X, columns=[col], drop_first=True)
  
  norm_method = st.selectbox("Méthode de normalisation", 
                ["Aucune", "StandardScaler", "MinMaxScaler", "Normalizer"])
  
  try:
    X = X.values
    
    if norm_method != "Aucune":
      with st.spinner("Application de la normalisation..."):
        if norm_method == "StandardScaler":
          scaler = StandardScaler()
          X = scaler.fit_transform(X)
          st.session_state.scaler = scaler
        elif norm_method == "MinMaxScaler":
          scaler = MinMaxScaler()
          X = scaler.fit_transform(X)
          st.session_state.scaler = scaler
        elif norm_method == "Normalizer":
          scaler = Normalizer()
          X = scaler.fit_transform(X)
          st.session_state.scaler = scaler
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if problem_type == "Classification":
      models = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Régression Logistique": LogisticRegression(max_iter=1000),
        "Arbre de Décision": DecisionTreeClassifier()
      }
      
      metrics = []
      for name, model in models.items():
        with st.spinner(f"Entraînement {name}..."):
          model.fit(X_train, y_train)
          y_pred = model.predict(X_test)
          
          metrics.append({
            "Modèle": name,
            "Précision": accuracy_score(y_test, y_pred),
            "Rappel": recall_score(y_test, y_pred, average='macro'),
            "F1-Score": f1_score(y_test, y_pred, average='macro'),
            "Précision (moy)": precision_score(y_test, y_pred, average='macro')
          })
          
          st.session_state.modeles_entraines[name] = model
      
      st.dataframe(pd.DataFrame(metrics).set_index("Modèle"))
      
    else:  # Régression
      models = {
        "Régression Linéaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR(),
        "Arbre de Décision": DecisionTreeRegressor()
      }
      
      metrics = []
      for name, model in models.items():
        with st.spinner(f"Entraînement {name}..."):
          model.fit(X_train, y_train)
          y_pred = model.predict(X_test)
          
          metrics.append({
            "Modèle": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R²": r2_score(y_test, y_pred)
          })
          
          st.session_state.modeles_entraines[name] = model
      
      st.dataframe(pd.DataFrame(metrics).set_index("Modèle"))
    
    st.success("Entraînement terminé avec succès!")
    
  except Exception as e:
    st.error(f"Erreur lors du traitement: {str(e)}")

# === 3. Prédiction ===
elif menu == "Prédiction":
  st.header("Prédiction")
  
  if not st.session_state.modeles_entraines:
    st.warning("Aucun modèle entraîné disponible.")
    st.stop()
  
  if not st.session_state.feature_names:
    st.warning("Informations sur les caractéristiques manquantes.")
    st.stop()
  
  df = st.session_state.data
  target = st.session_state.target
  features = [f for f in st.session_state.feature_names if f != target]
  
  model_name = st.selectbox("Modèle à utiliser", list(st.session_state.modeles_entraines.keys()))
  model = st.session_state.modeles_entraines[model_name]
  
  st.subheader("Valeurs d'entrée")
  input_values = []
  features = [f for f in st.session_state.feature_names if f != st.session_state.target]
  
  cols = st.columns(3)
  for i, feature in enumerate(features):
    with cols[i % 3]:
      min_val = float(df[feature].min())
      max_val = float(df[feature].max())
      mean_val = float(df[feature].mean())
      
      val = st.slider(
          label=feature,
          min_value=min_val,
          max_value=max_val,
          value=mean_val,
          step=(max_val - min_val)/100,
          key=f"slider_{i}"
      )
      input_values.append(val)
  
  if st.button("Exécuter la prédiction"):
    try:
      input_data = np.array(input_values).reshape(1, -1)
      
      if st.session_state.scaler:
        input_data = st.session_state.scaler.transform(input_data)
      
      prediction = model.predict(input_data)
      st.success(f"Résultat: {prediction[0]}")
      
      if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        st.metric("Probabilité", f"{max(proba)*100:.2f}%")
        
    except Exception as e:
      st.error(f"Erreur de prédiction: {str(e)}")