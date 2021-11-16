# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:15:37 2021

@author: em42363
"""
import sys
#sys.path.insert(0, r'C:\Users\eduar\OneDrive\PhD\UTuning')
#sys.path.insert(0, r'C:\Users\em42363\OneDrive\PhD\UTuning')

from UTuning import scorer, plots, UTuning

from catboost import CatBoostRegressor ## Decision-tree based gradient boosting
# Prediction model in the form of an ensemble of weak prediction models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

import numpy as np

#import os
#os.chdir(os.path.dirname(__file__))


df = pd.read_csv("https://raw.githubusercontent.com/emaldonadocruz/UTuning/master/dataset/unconv_MV.csv") #
#df = pd.read_csv(r'C:\Users\eduar\OneDrive\PhD\UTuning\dataset\unconv_MV.csv')
#df = pd.read_csv(r'C:\Users\em42363\OneDrive\PhD\UTuning\dataset\unconv_MV.csv')


# %% Split train test
'''
Perform split train test, and perform data min-max normalization
'''

y = df['Production'].values
X = df[['Por', 'LogPerm', 'Brittle', 'TOC']].values

scaler = MinMaxScaler()
scaler.fit(X)
Xs = scaler.transform(X)
ys = (y - y.min())/ (y.max()-y.min())

X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.33)

print(X_train.shape, y_train.shape)

# %% Model creation
'''
We define the model and the grid search space,
we pass the model and the grid search.
'''
n_estimators = np.arange(80, 150, step=10)
lr = np.arange(0.01, 0.15, step=.03)
param_grid = {
    "learning_rate": list(lr),
    "n_estimators": list(n_estimators)
}

model = CatBoostRegressor(loss_function='RMSEWithUncertainty',
                          verbose=False)

random_cv = UTuning.GridSearch(model, param_grid, 2)

random_cv.fit(X_train, y_train)
# %%Surface
'''
Similarly as in the problem with neural networks we can evaluate the
hyperparameter search space and use UTuning to construct the surface
'''
df = pd.DataFrame(random_cv.cv_results_)

labels = {'x': 'n estimators',
          'y': 'Learning rate',
          'z': 'Model goodness'}

plots.surface(df['param_n_estimators'],
              df['param_learning_rate'],
              (-1)*df['split0_test_score'],
              30,
              labels)


# %% Virtual ensemble
def virt_ensemble(X_train, y_train, num_samples=100, iters=1000, lr=0.2):
    ens_preds = []

    model = CatBoostRegressor(iterations=iters, learning_rate=lr, loss_function='RMSEWithUncertainty',
                              verbose=False, random_seed=0)

    model.fit(X_train, y_train)

    ens_preds = model.virtual_ensembles_predict(X_test, prediction_type='VirtEnsembles',
                                                virtual_ensembles_count=num_samples,
                                                thread_count=8)
    return np.asarray(ens_preds)


n_quantiles = 11
perc = np.linspace(0.0, 1.00, n_quantiles)

# %% Tuned model
'''
Next, we use the best parameters to construct and evaluate our model.
'''

Samples = 25

ens_preds = virt_ensemble(X_train,
                          y_train,
                          num_samples=Samples,
                          iters = random_cv.best_params_['n_estimators'],
                          lr = random_cv.best_params_['learning_rate'])

Pred_array = ens_preds[:, :, 0]

Knowledge_u = np.sqrt(np.var(Pred_array, axis=1))  # Knowledge uncertainty
Data_u = np.sqrt(np.mean(ens_preds[:, :, 1], axis=1))  # Data uncertainty
Sigma = Data_u+Knowledge_u

# %%
score = scorer.scorer(Pred_array, y_test, Sigma)

IF_array = score.IndicatorFunction()

# %% Second plot test
plots.error_accuracy_plot(perc, IF_array, Pred_array, y_test, Sigma)

# %%
print('Accuracy = {0:2.2f}'.format(score.Accuracy()))
print('Precision = {0:2.2f}'.format(score.Precision()))
print('Goodness = {0:2.2f}'.format(score.Goodness()))
