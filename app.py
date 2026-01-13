import streamlit as st
import pandas as pd
from src.clean_data import clean_data
from src.features import build_features
from src.dml_model import train_dml

st.title("Economics + AI: DML Causal Analysis")

st.markdown("""
This app estimates the **Average Treatment Effect (ATE)** of a treatment (e.g., minimum wage change) 
on an outcome (e.g., employment) using **Double Machine Learning (DML)**.
""")

# --- Upload CSV
uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data", df.head())

    df = clean_data(df)
    df = build_features(df)

    # --- Select columns
    outcome_col = st.selectbox("Select Outcome Column", df.columns)
    treatment_col = st.selectbox("Select Treatment Column", df.columns)
    control_cols = st.multiselect("Select Control Columns", df.columns)

    if st.button("Run DML"):
        Y = df[outcome_col]
        T = df[treatment_col]
        X = df[control_cols]

        model = train_dml(Y, T, X)
        ate = model.ate(X)

        st.success(f"Estimated Average Treatment Effect (ATE): {ate}")
