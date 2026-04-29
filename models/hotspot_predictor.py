import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def train_model(df):
    features = df[['NDVI', 'LST', 'urban_density']]
    labels = df['risk_score']
    model = DecisionTreeRegressor()
    model.fit(features, labels)
    return model
