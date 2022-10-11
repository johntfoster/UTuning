# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:27:35 2022

@author: em42363
"""

import os
os.chdir(os.path.dirname(__file__))

import sys
sys.path.insert(0, r'C:\Users\eduar\OneDrive\PhD\UTuning')
sys.path.insert(0, r'C:\Users\em42363\OneDrive\PhD\UTuning')

from UTuning import scorer, plots, UTSearch


from catboost import CatBoostRegressor ## Decision-tree based gradient boosting
# Prediction model in the form of an ensemble of weak prediction models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('https://raw.githubusercontent.com/emaldonadocruz/Publication_figure_style/master/Publication_figure_style.mplstyle')

df = pd.read_csv(r"C:\Users\em42363\OneDrive\PhD\Research\DTS_DTC_UncertaintyModels\train.csv")
df_test = pd.read_csv(r"C:\Users\em42363\OneDrive\PhD\Research\DTS_DTC_UncertaintyModels\test.csv")
df_test_results = pd.read_csv(r"C:\Users\em42363\OneDrive\PhD\Research\DTS_DTC_UncertaintyModels\Test_values.csv")
# %%

df.info()

# %% Split train test
'''
Perform split train test, and perform data min-max normalization
'''

y_train = df['DTC'].values
#y = df['DTS'].values
X_train = df[['CAL', 'CNC', 'GR', 'HRD', 'HRM', 'PE', 'ZDEN']].values

#y_test = df_test['DTC'].values
#y = df['DTS'].values
X_test = df_test[['CAL', 'CNC', 'GR', 'HRD', 'HRM', 'PE', 'ZDEN']].values
y_test = df_test_results['DTC'].values

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
y_train= (y_train- y_train.min())/ (y_train.max()-y_train.min())

scaler = MinMaxScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)
#y_test = (y_test - y_test.min())/ (y_test.max()-y_test.min())

#X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.33)


#print(X_train.shape, y_train.shape)

# %% Model creation
'''
We define the model and the grid search space,
we pass the model and the grid search.
'''
n_estimators = np.arange(200, 2200, step=200) #80 150
lr = np.arange(0.01, 0.08, step=.01) #0.1 0.15

param_grid = { 
    "learning_rate": list(lr),
    "n_estimators": list(n_estimators)
}

# %%
model = CatBoostRegressor(loss_function='RMSEWithUncertainty',
                          verbose=False)

random_cv = UTSearch.Grid(model, param_grid, 2)

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
              df['split0_test_score'],
              30,
              labels)
# %%
df = pd.DataFrame(random_cv.cv_results_)
#df.to_csv('Grid_search_well_logs_2_New.csv')


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

Samples = 50

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

# print('Accuracy = {0:2.2f}'.format(score.Accuracy()))
# print('Precision = {0:2.2f}'.format(score.Precision()))
# print('Goodness = {0:2.2f}'.format(score.Goodness()))


IF_array = score.IndicatorFunction()

# %% Second plot test
plots.error_accuracy_plot(perc, IF_array, Pred_array, y_test, Sigma)

# %%
print('Accuracy = {0:2.2f}'.format(score.Accuracy()))
print('Precision = {0:2.2f}'.format(score.Precision()))
print('Goodness = {0:2.2f}'.format(score.Goodness()))
