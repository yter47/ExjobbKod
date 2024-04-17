import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_original=pd.read_csv("C:/Skola/Examensarbete/footballData.csv")
sample = ['potential', 'league_rank', 'overall', 'age', 'value_eur']
df = df_original[sample].copy()
df.head()
print(df.shape)
df.info()
print(df.describe)

sample = ['potential', 'league_rank', 'overall', 'age']