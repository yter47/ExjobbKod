import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from contextlib import contextmanager
from time import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read data
data0 = pd.read_csv("C:/Skola/Examensarbete/footballData.csv")

from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
def labelencoder(df):
    for c in df.columns:
        if df[c].dtype == 'object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df

data1 = labelencoder(data0)

# Split data into train and test sets
m = len(data1)
M = list(range(m))
random.seed(2021)
random.shuffle(M)

train = data1.iloc[M[0:(m//4)*3]]
test = data1.iloc[M[(m//4)*3:]]

# Define features and target
sample = ['potential', 'league_rank', 'overall', 'age', 'player_positions']
target = ['value_eur']
trainY = train[target]
trainX = train[sample]
testY = test[target]
testX = test[sample]

from sklearn.impute import SimpleImputer

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
trainX_imputed = imputer.fit_transform(trainX)
testX_imputed = imputer.transform(testX)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(trainX_imputed, trainY)

# Predict on test data
y_pred = model.predict(testX_imputed)

# Calculate RMSE and R² score
rmse = np.sqrt(mean_squared_error(testY, y_pred))
r2 = r2_score(testY, y_pred)

print('RMSE: ', rmse)
print('R² score: ', r2)

# Plot real vs predicted values
plt.plot(testY.values.ravel(), label='Real')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
