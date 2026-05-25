import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):

    columns = [
        'PM2.5',
        'PM10',
        'SO2',
        'NO2',
        'CO',
        'O3',
        'TEMP',
        'PRES',
        'DEWP',
        'RAIN'
    ]

    df = df[columns].copy()

    imputer = SimpleImputer(strategy='mean')
    df_imputed = imputer.fit_transform(df)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_imputed)

    return scaled