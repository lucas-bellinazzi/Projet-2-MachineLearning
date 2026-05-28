import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

st.set_page_config(page_title="Project AI1", layout="wide")

if 'data' not in st.session_state:
  st.session_state.data = None
if 'trained_models' not in st.session_state:
  st.session_state.trained_models = {}
if 'scaler' not in st.session_state:
  st.session_state.scaler = None
if 'feature_names' not in st.session_state:
  st.session_state.feature_names = []
if 'target' not in st.session_state:
  st.session_state.target = ''

st.sidebar.title("Navigation")
menu = st.sidebar.radio("", ["Upload", "Machine Learning", "Prediction"], label_visibility="collapsed")

if menu == "Upload":
  st.header("File Upload")
  uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
  if uploaded_file is not None:
    try:
      if "pima" in uploaded_file.name.lower():
        col=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(uploaded_file, names=col)
      elif "housing" in uploaded_file.name.lower():
        col = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
        df = pd.read_csv(uploaded_file, names=col, delim_whitespace=True)
      else:
        df = pd.read_csv(uploaded_file)

      if df.empty:
        st.error("Error: The file seems to be empty.")
      else:
        st.session_state.data = df
        st.success(f"File loaded successfully! Shape: {df.shape}")
        st.markdown("## Data Preview")
        st.dataframe(df)
        st.markdown("## Descriptive Statistics")
        st.write(df.describe())

        st.markdown("## Histograms")
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
          plt.suptitle("Feature Histograms", y=1.02)
          plt.tight_layout()
          st.pyplot(fig)
        except Exception as e:
          st.error(f"Error generating histograms: {str(e)}")

        st.session_state.trained_models = {}
        st.session_state.scaler = None
        st.session_state.feature_names = df.columns.tolist()
    except Exception as e:
      st.error(f"Read error: {str(e)}")

elif menu == "Machine Learning":
  st.header("Machine Learning")
  if st.session_state.data is None:
    st.warning("Please upload a CSV file first.")
    st.stop()

  df = st.session_state.data
  if df.empty:
    st.error("Error: The data is empty.")
    st.stop()

  col1, col2 = st.columns(2)
  with col1:
    problem_type = st.radio("Problem Type", ["Classification", "Regression"])
  with col2:
    target_var = st.selectbox("Target Variable", df.columns)

  st.session_state.target = target_var
  X = df.drop(columns=[target_var])
  y = df[target_var]

  for col in X.select_dtypes(include=['object']).columns:
    try:
      X[col] = pd.to_numeric(X[col], errors='raise')
    except:
      X = pd.get_dummies(X, columns=[col], drop_first=True)

  norm_method = st.selectbox("Normalization Method", ["None", "StandardScaler", "MinMaxScaler", "Normalizer"])

  try:
    X = X.values
    if norm_method != "None":
      with st.spinner("Applying normalization..."):
        if norm_method == "StandardScaler":
          scaler = StandardScaler()
        elif norm_method == "MinMaxScaler":
          scaler = MinMaxScaler()
        elif norm_method == "Normalizer":
          scaler = Normalizer()
        X = scaler.fit_transform(X)
        st.session_state.scaler = scaler

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []

    if problem_type == "Classification":
      models = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier()
      }
      for name, model in models.items():
        with st.spinner(f"Training {name}..."):
          model.fit(X_train, y_train)
          y_pred = model.predict(X_test)
          results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred, average='macro'),
            "F1-Score": f1_score(y_test, y_pred, average='macro'),
            "Avg Precision": precision_score(y_test, y_pred, average='macro')
          })
          st.session_state.trained_models[name] = model
      df_res = pd.DataFrame(results)
      st.dataframe(df_res.set_index("Model"))
      for metric in ['Accuracy', 'Recall', 'F1-Score', 'Avg Precision']:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=df_res, x='Model', y=metric, ax=ax)
        ax.set_title(f'Model Comparison - {metric}')
        st.pyplot(fig)
    else:
      models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "SVR": SVR(),
        "Decision Tree": DecisionTreeRegressor()
      }
      for name, model in models.items():
        with st.spinner(f"Training {name}..."):
          model.fit(X_train, y_train)
          y_pred = model.predict(X_test)
          results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R²": r2_score(y_test, y_pred)
          })
          st.session_state.trained_models[name] = model
      df_res = pd.DataFrame(results)
      st.dataframe(df_res.set_index("Model"))
      for metric in ['MAE', 'MSE', 'R²']:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=df_res, x='Model', y=metric, ax=ax)
        ax.set_title(f'Model Comparison - {metric}')
        st.pyplot(fig)
    st.success("Training completed successfully!")
  except Exception as e:
    st.error(f"Error during processing: {str(e)}")

elif menu == "Prediction":
  st.header("Prediction")
  if not st.session_state.trained_models:
    st.warning("No trained models available.")
    st.stop()
  if not st.session_state.feature_names:
    st.warning("Missing feature information.")
    st.stop()

  df = st.session_state.data
  target = st.session_state.target
  features = [f for f in st.session_state.feature_names if f != target]
  model_name = st.selectbox("Model to use", list(st.session_state.trained_models.keys()))
  model = st.session_state.trained_models[model_name]

  st.subheader("Input Values")
  input_values = []
  cols = st.columns(3)
  for i, feature in enumerate(features):
    with cols[i % 3]:
      min_val = float(df[feature].min())
      max_val = float(df[feature].max())
      mean_val = float(df[feature].mean())
      val = st.slider(label=feature, min_value=min_val, max_value=max_val, value=mean_val, step=(max_val - min_val)/100, key=f"slider_{i}")
      input_values.append(val)

  if st.button("Run Prediction"):
    try:
      input_data = np.array(input_values).reshape(1, -1)
      if st.session_state.scaler:
        input_data = st.session_state.scaler.transform(input_data)
      prediction = model.predict(input_data)
      st.success(f"Result: {prediction[0]}")
      if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        st.metric("Probability", f"{max(proba)*100:.2f}%")
    except Exception as e:
      st.error(f"Prediction error: {str(e)}")
