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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from IPython.display import display


data0=pd.read_csv("C:/Skola/Examensarbete/footballData.csv")

from sklearn.preprocessing import LabelEncoder

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df

data1=labelencoder(data0)

m=len(data1)
M=list(range(m))
random.seed(2021)
random.shuffle(M)

train=data1.iloc[M[0:(m//4)*3]]
test=data1.iloc[M[(m//4)*3:]]

sample = ['potential', 'league_rank', 'overall', 'age', 'player_positions']
target=['value_eur']
trainY=train[target]
trainX=train[sample]
testY=train[target]
testX=train[sample]
df_columns = list(trainX.columns)

print(df_columns)
print('x_train shape: ', trainX.shape, ' y_train shape: ', trainY.shape)
print('x_test shape: ', testX.shape, ' y_test shape: ', testY.shape)

x_train = trainX.values
x_test = testX.values

# ravel() används för att "platta till" data till en endimensionell array.
train_data = lgbm.Dataset(x_train, label=trainY.values.ravel())  
eval_data = lgbm.Dataset(x_test, label=testY.values.ravel(), reference=train_data) 

# Längden av x och y träningsdata
print('Train data x length: ', len(train_data.data))
print('Train data y length: ', len(train_data.label))

# Längden av x och y test data
print('Test data x length: ', len(eval_data.data))
print('Test data y length: ', len(eval_data.label))

params = {
    'learning_rate':0.1,
    'boosting_type':'gbdt', 
    'objective':'regression', 
    'metric':'rmse', 
    'max_depth':5
    }

model = lgbm.train(params, train_data, valid_sets=[eval_data], num_boost_round=500)
model.save_model('lightGBM_model.txt')

y_pred = model.predict(x_test, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(testY, y_pred))
# R² score
r2 = r2_score(testY, y_pred)

print('RMSE: ', rmse)
print('R² score: ', r2)

plt.plot(testY.values.ravel(), label='Real')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()