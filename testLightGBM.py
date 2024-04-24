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


with pd.option_context('mode.use_inf_as_na', True):
    data0 = pd.read_csv("C:/Skola/Examensarbete/footballData.csv")
display(data0[0:3].T)
data0[['value_eur']].info()

data1 = data0.copy()

from sklearn.preprocessing import LabelEncoder

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df

#swedish_allsvenskan_data = data0[data0['league_name'] == 'Swedish Allsvenskan']
#data1=labelencoder(swedish_allsvenskan_data)
data1=labelencoder(data0)
m=len(data1)
M=list(range(m))
random.seed(2021)
random.shuffle(M)

train=data1.iloc[M[0:(m//4)*3]]
test=data1.iloc[M[(m//4)*3:]]

target=['value_eur']
#testAllsvenskan=['league_name']
#trainY=train[target]
#trainX=train.drop(target,axis=1)
#trainX=train[testAllsvenskan]
#testY=train[target]
#testX=train.drop(target,axis=1)
#testX=train[testAllsvenskan]

print(data0.head())

print('Start')
print(data0['league_name'].unique())
print('End')

# Filtrera data för Swedish Allsvenskan
#swedish_allsvenskan_data = data0[data0['league_name'] == 'Swedish Allsvenskan']
attributes = ['potential', 'age', 'overall', 'player_positions', 'international_reputation']
# Definiera attribut och målvariabler för träningsuppsättningen
trainY = train[target]
#trainX = train.drop(target, axis=1)
trainX = train[attributes]

# Definiera attribut och målvariabler för testuppsättningen
testY = train[target]
#testX = train.drop(target, axis=1)
testX = train[attributes]

df_columns = list(trainX.columns)
print(df_columns)

def create_numeric_feature(input_df):
    use_columns = df_columns 
    return input_df[use_columns].copy()

from contextlib import contextmanager
from time import time

class Timer:
    def __init__(self, logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None, sep=' '):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)

from tqdm import tqdm

def to_feature(input_df):
    processors = [
        create_numeric_feature
    ]
    
    out_df = pd.DataFrame()
    
    for func in tqdm(processors, total=len(processors)):
        with Timer(prefix='create' + func.__name__ + ' '):
            _df = func(input_df)

        assert len(_df) == len(input_df), func.__name__
        out_df = pd.concat([out_df, _df], axis=1)
        
    return out_df

train_feat_df = to_feature(trainX)
test_feat_df = to_feature(testX)
    
def fit_lgbm(X, y, cv, params: dict=None, verbose: int=50):
    if params is None:
        params = {}
        
    models = []
    oof_pred = np.zeros_like(y, dtype=np.float64)

    for i, (idx_train, idx_valid) in enumerate(cv): 
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMRegressor(**params)
        
        with Timer(prefix='fit fold={} '.format(i)):
            clf.fit(x_train, y_train, 
                    eval_set=[(x_valid, y_valid)])
        
        pred_i = clf.predict(x_valid)
        oof_pred[idx_valid] = pred_i
        models.append(clf)
        mse = mean_squared_error(y_valid, pred_i)  # Calculate MSE
        rmse = np.sqrt(mse)  # Calculate RMSE from MSE
        print(f'Fold {i} RMSE: {rmse:.4f}')  # Print RMSE
        print()
        
    score = mean_squared_error(y, oof_pred) ** .5
    print('-' * 50)
    print('FINISHED | Whole RMSE: {:.4f}'.format(score))
    return oof_pred, models

params = {
    'objective': 'rmse', 
    'learning_rate': .05,
    'reg_lambda': .1,
    'reg_alpha': .1,
    'max_depth': -1, 
    'n_estimators': 10000, 
    'colsample_bytree': .5, 
    'min_child_samples': 10,
    'subsample_freq': 3,
    'subsample': .9,
    'importance_type': 'gain', 
    'random_state': 71,
    'num_leaves': 62,
    'verbosity': -1
}

ydf=trainY

from sklearn.model_selection import KFold

MODELS = []
for i, target_col in enumerate(target):
    fold = KFold(n_splits=5, shuffle=True, random_state=71)
    ydfi = ydf[target_col]
    y=np.array(ydfi)
    cv = list(fold.split(train_feat_df, y))
    oof, models = fit_lgbm(train_feat_df.values, y, cv, params=params)
    MODELS+=[models]
    
    fig,ax = plt.subplots(figsize=(6,6))
    ax.set_title(target_col,fontsize=20)
    ax.set_ylabel('Train Predicted '+target_col,fontsize=12)
    ax.set_xlabel('Train Actual '+target_col,fontsize=12)
    ax.scatter(y,oof)
        
    
def visualize_importance(models, feat_train_df):
     
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importances_
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)
    
    order = feature_importance_df.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:50]
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
    
    sns.boxenplot(data=feature_importance_df, x='feature_importance', y='column', order=order, 
        ax=ax, palette='viridis', orient='h')
    
    ax.tick_params(axis='x', rotation=0)
    #ax.set_title('Importance')
    ax.grid()
    fig.tight_layout()
    
    return fig,ax

for i, target_col in enumerate(target):
    models=MODELS[i]
    fold = KFold(n_splits=5, shuffle=True, random_state=71)
    ydfi=ydf[target_col]
    y=np.array(ydfi)
    cv = list(fold.split(train_feat_df, y))
    oof, models = fit_lgbm(train_feat_df.values, y, cv, params=params)
    fig, ax = visualize_importance(models, train_feat_df)
    ax.set_title(target_col+' Imortance',fontsize=20)