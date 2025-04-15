from pathlib import Path
import pandas as pd
from math import sqrt

def get_dataset(db_path):  # TODO:move to common module
    df = pd.read_csv(db_path)
    dataset = df.values.tolist()
    print(f"The Dataset:\n{dataset}")
    return dataset