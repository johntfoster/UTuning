#!/usr/bin/env python
# coding: utf-8

# # Catboost regression optmized by Model Goodness

# ## Package import

# In[1]: Import functions

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns
import numpy as np

import os
os.chdir(os.path.dirname(__file__))

import sys
#sys.path.insert(0, r'C:\Users\eduar\OneDrive\PhD\UTuning')
sys.path.insert(0, r'C:\Users\em42363\OneDrive\PhD\UTuning')

from UTuning import scorer, plots

#df = pd.read_csv(r'C:\Users\eduar\OneDrive\PhD\UTuning\dataset\unconv_MV.csv')
df = pd.read_csv(r'C:\Users\em42363\OneDrive\PhD\UTuning\dataset\unconv_MV.csv')

import random
import matplotlib.pyplot as plt

# In[1]: Split train test

y = df['Production'].values
X = df[['Por', 'LogPerm', 'Brittle', 'TOC']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print(X_train.shape, y_train.shape)


# In[6]: Regressor

model = CatBoostRegressor(iterations=1000, learning_rate=0.2, loss_function='RMSEWithUncertainty',
                          verbose=False, random_seed=0)
model.fit(X_train, y_train)


estimates = model.predict(X_test)
print(estimates.shape)

# In[9]: Plot error line

plots.error_line(estimates[:, 0], y_test, np.sqrt(estimates[:, 1]), Frac=1)

# %%
def virt_ensemble(X_train,y_train, num_samples=100, iters=1000, lr=0.2):
    ens_preds = []
    
    model = CatBoostRegressor(iterations=iters, learning_rate=lr, loss_function='RMSEWithUncertainty',
                          verbose=False, random_seed=0)



    model.fit(X_train,y_train)

    
    ens_preds = model.virtual_ensembles_predict(X_test, prediction_type='VirtEnsembles', 
                                                virtual_ensembles_count=num_samples,
                                                thread_count=8)
    return np.asarray(ens_preds)

# %%
n_quantiles = 11
perc = np.linspace(0.0, 1.00, n_quantiles)

Samples = 10

ens_preds=virt_ensemble(X_train,y_train, num_samples=Samples)

Pred_array = ens_preds[:,:,0]

Knowledge_u=np.sqrt(np.var(Pred_array,axis=1)) #Knowledge uncertainty
Data_u=np.sqrt(np.mean(ens_preds[:,:,1],axis=1)) #Data uncertainty
Sigma=Knowledge_u+Data_u

# %%
scorer = scorer.scorer(Pred_array, y_test, Sigma)

IF_array = scorer.IndicatorFunction()

# %% Second plot test
plots.error_accuracy_plot(perc,IF_array,Pred_array,y_test,Sigma)

# %%
print('Accuracy = {0:2.2f}'.format(scorer.Accuracy()))
print('Precision = {0:2.2f}'.format(scorer.Precision()))
print('Goodness = {0:2.2f}'.format(scorer.Goodness()))
