import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(subset=["question", "answer"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
