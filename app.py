import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.metrics import r2_score
import os

st.set_page_config(page_title="Bank Analytics Dashboard", layout="wide")
st.title("📊 Bank Profitability Dashboard")

st.write("App started successfully ✅")

# Load CSV safely
try:
    file_path = os.path.join(os.path.dirname(__file__), "Indian_Banks.csv")
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.replace('\n', ' ', regex=True).str.strip()

    st.write("Data Loaded ✅")
except Exception as e:
    st.error(f"CSV Error: {e}")
    st.stop()

# Load model safely
try:
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    model = pickle.load(open(model_path, "rb"))
    st.write("Model Loaded ✅")
except Exception as e:
    st.error(f"Model Error: {e}")
    st.stop()

# Define features
features = [
    "Net NPA Ratio (%)",
    "CAR / CRAR (%)",
    "Credit Growth (%)",
    "Cost-to- Income Ratio (%)",
    "Bank Size [Log(Assets)]",
    "NPA × CAR [Interaction]",
    "NPA × Size [Interaction]"
]

# Check columns
st.write("Columns:", df.columns)

try:
    X = df[features]
    y = df["ROA (%) [DV]"]
except Exception as e:
    st.error(f"Column Error: {e}")
    st.stop()

# Model prediction
try:
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()

# Dashboard
banks = st.multiselect("Select Banks", df["Bank Name"].unique(),
                       default=df["Bank Name"].unique()[:2])

filtered_df = df[df["Bank Name"].isin(banks)]

col1, col2, col3 = st.columns(3)
col1.metric("Avg ROA", round(filtered_df["ROA (%) [DV]"].mean(), 2))
col2.metric("Avg NPA", round(filtered_df["Net NPA Ratio (%)"].mean(), 2))
col3.metric("R² Score", round(r2, 3))

# Chart
fig = px.line(filtered_df, x="Year", y="ROA (%) [DV]",
              color="Bank Name", markers=True)
st.plotly_chart(fig)

# Prediction
st.subheader("🔮 ROA Prediction")

npa = st.number_input("NPA")
car = st.number_input("CAR")
credit = st.number_input("Credit Growth")
cost_income = st.number_input("Cost Income")
size = st.number_input("Size")

npa_car = npa * car
npa_size = npa * size

if st.button("Predict ROA"):
    try:
        pred = model.predict([[npa, car, credit, cost_income, size, npa_car, npa_size]])
        st.success(f"Predicted ROA: {pred[0]}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
