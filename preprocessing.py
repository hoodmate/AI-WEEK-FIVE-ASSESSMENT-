import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess(df):
    df = df.dropna()
    return df
