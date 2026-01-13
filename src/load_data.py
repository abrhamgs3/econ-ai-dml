import pandas as pd
from pathlib import Path

DATA_RAW = Path("data/raw")

def load_raw_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / filename)
