# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:15:37 2021

@author: em42363
"""


from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

import numpy as np

import os
os.chdir(os.path.dirname(__file__))

import sys
sys.path.insert(0, r'C:\Users\eduar\OneDrive\PhD\UTuning')
sys.path.insert(0, r'C:\Users\em42363\OneDrive\PhD\UTuning')

from UTuning import scorer, plots, UTuning

#df = pd.read_csv(r'C:\Users\eduar\OneDrive\PhD\UTuning\dataset\unconv_MV.csv')
df = pd.read_csv(r'C:\Users\em42363\OneDrive\PhD\UTuning\dataset\unconv_MV.csv')


# In[1]: Split train test

y = df['Production'].values
X = df[['Por', 'LogPerm', 'Brittle', 'TOC']].values

scaler = MinMaxScaler()
scaler.fit(X)
Xs = scaler.transform(X)

ys = (y - y.min())/ (y.max()-y.min())

X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.33)

#%%
n_estimators = np.arange(90, 200, step=10)
lr = np.arange(0.001, 0.2, step=.001)
param_grid = {
    "learning_rate": list(lr),
    "n_estimators": list(n_estimators)
}

model=CatBoostRegressor(loss_function='RMSEWithUncertainty',
                        verbose=False,
                        random_seed=0)

random_cv = UTuning.RandomizedSearch(model, param_grid, cv = 2, n_iter = 25)

random_cv.fit(X_train, y_train)
#%% surface

df = pd.DataFrame(random_cv.cv_results_)

labels = {'x': 'n estimators',
          'y': 'Learning rate',
          'z': 'Model goodness'}

x = np.array(df['param_n_estimators'], dtype = float)
y = np.array(df['param_learning_rate'], dtype = float)
z = np.array(df['split0_test_score'], dtype = float)

plots.surface(x,
              y,
              z,
              10,
              labels)


# %% Tuned model
model = CatBoostRegressor(iterations=random_cv.best_params_['n_estimators'],
                          learning_rate=random_cv.best_params_['learning_rate'],
                          loss_function='RMSEWithUncertainty',
                          verbose=False, random_seed=0)

model.fit(X_train, y_train)

estimates = model.predict(X_test)

#%%
n_quantiles = 11
perc = np.linspace(0.0, 1.00, n_quantiles)

Prediction = estimates[:, 0]

#Knowledge_u = np.sqrt(np.var(Prediction, axis=1))  # Knowledge uncertainty
#Data_u = np.sqrt(np.mean(ens_preds[:, :, 1], axis=1))  # Data uncertainty
#Sigma = Knowledge_u+Data_u
Sigma = np.sqrt(estimates[:,1])

score = scorer.scorer(Prediction, y_test, Sigma)

#%%
IF_array = score.IndicatorFunction()


plots.error_accuracy_plot(perc, IF_array, Prediction, y_test, Sigma)


print('Accuracy = {0:2.2f}'.format(score.Accuracy()))
print('Precision = {0:2.2f}'.format(score.Precision()))
print('Goodness = {0:2.2f}'.format(score.Goodness()))
