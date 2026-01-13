import pandas as pd
from src.clean_data import clean_data
from src.features import build_features
from src.dml_model import train_dml

def main():
    # --- Synthetic dataset for smoke test
    df = pd.DataFrame({
        "employment": [1, 0, 1, 1, 0, 1, 0, 1],
        "min_wage_change": [0.5, 0.5, 0, 0, 1, 1, 0, 0],
        "age": [22, 19, 35, 40, 18, 25, 50, 30],
        "education": [12, 11, 16, 16, 10, 14, 12, 15]
    })

    df = clean_data(df)
    df = build_features(df)

    Y = df["employment"]
    T = df["min_wage_change"]
    X = df[["age", "education"]]

    # Train DML
    dml_model = train_dml(Y, T, X)

    # Average Treatment Effect
    ate = dml_model.ate(X)
    print(f"Estimated ATE: {ate}")

if __name__ == "__main__":
    main()
