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
import lightgbm as lgbm
import category_encoders as ce
#from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from IPython.display import display


df_original=pd.read_csv("C:/Skola/Examensarbete/footballData.csv")
sample = ['potential', 'league_rank', 'overall', 'age', 'value_eur']
df = df_original[sample].copy()
df.head()
print(df.shape)
df.info()
print(df.describe)

sample = ['potential', 'league_rank', 'overall', 'age']

for i, col in enumerate(sample):
    grouped_data = df[[col, 'value_eur']].groupby(col).mean()
    print(grouped_data)
    print()

plt.subplots(figsize=(15, 15))
sample = ['potential', 'league_rank', 'overall', 'age']
for i, col in enumerate(sample):
    # Create subplots in a 3x2 grid
    plt.subplot(3, 2, i + 1)
     
    # Create a countplot for the current column
    sns.countplot(data=df, x=col)
     
# Adjust subplot layout for better presentation
plt.tight_layout()
 
# Display the subplots
plt.show()

features = df.drop('value_eur', axis=1)

target = df['value_eur']

X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                  random_state=2023,
                                                  test_size=0.25)

print(X_train.shape, X_val.shape)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)


train_data = lgbm.Dataset(X_train, label=Y_train)
test_data = lgbm.Dataset(X_val, label=Y_val, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.15,
    'feature_fraction': 0.5,
}

num_round = 1000
bst = lgbm.train(params, train_data, num_round, valid_sets=[test_data])


from sklearn.metrics import mean_squared_error as mse
from lightgbm import LGBMRegressor
 
# Create an instance of the LightGBM Regressor with the RMSE metric.
model = LGBMRegressor(metric='rmse')
 
# Train the model using the training data.
model.fit(X_train, Y_train)
 
# Make predictions on the training and validation data.
y_train = model.predict(X_train)
y_val = model.predict(X_val)

print("Training RMSE: ", np.sqrt(mse(Y_train, y_train)))
print("Validation RMSE: ", np.sqrt(mse(Y_val, y_val)))