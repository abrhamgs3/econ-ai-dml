import streamlit as st
import pandas as pd
import numpy as np
from math import erf, sqrt

from src.clean_data import clean_data
from src.features import build_features
from src.dml_model import train_dml

st.title("Economics + AI: DML Causal Analysis")

st.markdown("""
This app estimates the **Average Treatment Effect (ATE)** of a treatment (e.g., minimum wage change)
on an outcome (e.g., employment) using **Double Machine Learning (DML)**.
It also reports standard errors, a 95% confidence interval, an unadjusted comparison,
charts of the raw patterns in the data, and how the estimated effect varies across observations.
""")

# --- Upload CSV
uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Data preview")
    st.write(df.head())
    st.write(
        f"Rows: {len(df)} | Columns: {len(df.columns)} | "
        f"Missing cells: {int(df.isna().sum().sum())}"
    )

    df = clean_data(df)
    df = build_features(df)

    st.subheader("2. Summary statistics")
    st.dataframe(df.describe(include="all").T)

    # --- Select columns
    st.subheader("3. Choose variables")
    outcome_col = st.selectbox("Select Outcome Column", df.columns)
    treatment_col = st.selectbox("Select Treatment Column", df.columns)
    control_cols = st.multiselect("Select Control Columns", df.columns)

    if st.button("Run DML"):
        if not control_cols:
            st.error("Select at least one control column before running DML.")
        else:
            Y = df[outcome_col]
            T = df[treatment_col]
            X = df[control_cols]

            st.subheader("4. Unadjusted comparison (before DML)")
            unadj = df.groupby(treatment_col)[outcome_col].agg(["mean", "count"])
            unadj.columns = [f"Mean of {outcome_col}", "Observations"]
            st.dataframe(unadj)
            st.bar_chart(unadj[[f"Mean of {outcome_col}"]])
            if T.nunique() == 2:
                tvals = sorted(T.unique())
                naive = Y[T == tvals[1]].mean() - Y[T == tvals[0]].mean()
                st.write(f"Naive difference in means (no controls): {naive:.4f}")

            st.write("Average control values by treatment level (are the groups comparable?)")
            balance = df.groupby(treatment_col)[control_cols].mean().round(3)
            st.dataframe(balance)
            st.bar_chart(balance)

            # --- DML with inference
            model = train_dml(Y, T, X)

            st.subheader("5. DML results")
            ate = float(model.ate(X))
            inf = model.ate_inference(X)
            se = float(inf.stderr_mean)
            ci_low, ci_high = (float(v) for v in inf.conf_int_mean(alpha=0.05))
            z = ate / se if se > 0 else float("nan")
            p = (
                2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
                if se > 0
                else float("nan")
            )

            results = pd.DataFrame(
                {
                    "Estimate (ATE)": [ate],
                    "Std. error": [se],
                    "z": [z],
                    "p-value": [p],
                    "95% CI lower": [ci_low],
                    "95% CI upper": [ci_high],
                }
            )
            st.dataframe(results.round(4))
            st.success(
                f"Estimated Average Treatment Effect (ATE): {ate:.4f} "
                f"(95% CI: {ci_low:.4f} to {ci_high:.4f})"
            )

            st.subheader("6. How the effect varies across observations")
            effects = pd.Series(
                np.asarray(model.effect(X)).ravel(),
                index=df.index,
                name="Estimated effect",
            )
            st.dataframe(effects.describe().to_frame().T.round(4))
            counts, _ = np.histogram(effects, bins=10)
            hist = pd.DataFrame(
                {"Observations": counts},
                index=[f"Bin {i + 1}" for i in range(len(counts))],
            )
            st.bar_chart(hist)

            st.subheader("7. Average effect by subgroup")
            for col in control_cols:
                try:
                    if df[col].nunique() <= 5:
                        groups = df[col]
                        label = f"{col} group"
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        groups = pd.qcut(df[col], 4, duplicates="drop")
                        label = f"{col} quartile"
                    else:
                        continue
                    sub = effects.groupby(groups, observed=True).mean()
                    sub.index = sub.index.astype(str)
                    st.write(f"Average effect by {label}")
                    st.bar_chart(sub.to_frame("Average effect"))
                except Exception:
                    continue

            numeric_controls = [
                c for c in control_cols if pd.api.types.is_numeric_dtype(df[c])
            ]
            if numeric_controls:
                col = numeric_controls[0]
                scatter = pd.DataFrame({col: df[col], "Estimated effect": effects})
                st.write(f"Estimated effect against {col}")
                st.scatter_chart(scatter, x=col, y="Estimated effect")
