# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:05:43 2021

@author: Eduardo Maldonado Cruz emaldoandocruz@utexas.edu
"""

#from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

import pandas as pd

import numpy as np

import os
os.chdir(os.path.dirname(__file__))

import sys
#sys.path.insert(0, r'C:\Users\eduar\OneDrive\PhD\UTuning')
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

#print(X_train.shape, y_train.shape)

# %%
from tensorflow import keras

from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.layers.core import Lambda
from keras import backend as K

def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))

def build_model(activation = 'relu',
                dropout_rate = 0.2,
                lr = 0.01):
    model = Sequential()
    
    model.add(Flatten(input_shape = [X.shape[1]]))
    model.add(Dense(32, activation=activation))
    model.add(PermaDropout(dropout_rate))
    model.add(Dense(32, activation=activation))
    model.add(PermaDropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(loss = 'mean_squared_error',
                  optimizer = keras.optimizers.Adam(0.001))
    
    return model

model = KerasRegressor(build_fn=build_model, verbose=0)


model.fit(X_train,y_train, epochs = 1000)
#%%

def ensemble(model, X_s, batch_size,y_s):
    #Take n_samples to draw a distribution
    n_samples = 50
    mc_predictions = np.zeros((n_samples, y_s.shape[0]))
    for i in range(n_samples):
        #y_p = mc_model.predict(X_test, batch_size=4)
        y_p = model.predict(X_s, verbose=1, batch_size=batch_size)
        mc_predictions[i] = (y_p)
    return mc_predictions

# %%
mc_predictions = ensemble(model, X_test, 8, y_test)
# %%
Sigma = np.std(mc_predictions , axis = 0)
Pred_array = mc_predictions
Pred_array = Pred_array.T
# # %%
score = scorer.scorer(Pred_array, y_test, Sigma)

print('Accuracy = {0:2.2f}'.format(score.Accuracy()))
print('Precision = {0:2.2f}'.format(score.Precision()))
print('Goodness = {0:2.2f}'.format(score.Goodness()))

# %%
n_quantiles = 11
perc = np.linspace(0.0, 1.00, n_quantiles)

IF_array = score.IndicatorFunction()
avgIF = np.mean(IF_array,axis=0)

plots.error_accuracy_plot(perc,IF_array,Pred_array,y_test,Sigma)

#%%
lr = [0.1, 0.01, 0.001, 0.0001]
dropout_rate = [0.1, 0.2, 0.3, 0.4]
param_grid = dict(lr=lr, dropout_rate=dropout_rate)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, scoring = 'neg_mean_absolute_error')
grid = UTuning.GridSearchKeras(model, param_grid)
grid_result = grid.fit(X_train, y_train)

# %%
df = pd.DataFrame(grid_result.cv_results_)

labels = {'x': 'Learning rate',
          'y': 'Dropout rate',
          'z': 'Model goodness'}

plots.surface(df['param_lr'],
              df['param_dropout_rate'],
              -1*(df['split0_test_score']),
              50,
              labels)
