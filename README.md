# Hyperparameter Uncertainty Tuning

Uncertainty Tuning (UTuning) is a package that focuses on summarizing uncertainty model performance for optimum hyperparameter tuning.

This library uses the metric proposed by Maldonado-Cruz and Pyrcz (2021) to tune model hyperparameters based on the uncertainty model goodness metric.
Maldonado-Cruz, E., & Pyrcz, M. J. (2021). Tuning machine learning dropout for subsurface uncertainty model accuracy. Journal of Petroleum Science and Engineering, 205, 108975. https://doi.org/https://doi.org/10.1016/j.petrol.2021.108975

<p align="center">
    <img src="https://raw.githubusercontent.com/emaldonadocruz/UTuning/master/figures/CrossVal.png"/>
</p>

In the figure we show a comparison of the cross-validation plot and respective accuracy plot for two uncertainty models where the hyperparameters were optimized using different objective functions. a) Using MAE, b) Uncertainty model goodness.
Both models have a high Pearson's correlation coefficient yet model in b) is a better uncertainty model.

## Features
This is what UTuning has to offer:

* Hyperparameter tuning for ensemble based uncertainty models
* Robust uncertainty evaluation
* Evaluation of uncertainty models

## Installation

### Dependencies

- numpy (>=1.16)
- scikit-learn (>=0.23)

### User Installation

`pip install UTuning`

## Examples

### Tune Machine Learning uncertainty model with GridSearchCV
In this first example we use Catboost as ensemble learner for predictions of production.

For this notebook example we have a problem that consists on predicting **permeability** from **porosity** and **acoustic impedance data**. We have selected this problem because we are primarily interested in capturing the uncertainty related to predictions of permeability based on existing data. 
This problem can be expanded to any prediction problem.

To start out, change our import statement to get UTuning grid search cross validation interface, and the rest is almost identical!

```python
"""
Created on Mon Sep 20 16:15:37 2021
@author: em42363
"""
from UTuning import scorer, plots, UTSearch

from catboost import CatBoostRegressor ## Decision-tree based gradient boosting
# Prediction model in the form of an ensemble of weak prediction models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/emaldonadocruz/UTuning/master/dataset/unconv_MV.csv") #

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
n_estimators = np.arange(180, 220, step=1) #80 150
lr = np.arange(0.035, 0.06, step=.001) #0.1 0.15
param_grid = {
    "learning_rate": list(lr),
    "n_estimators": list(n_estimators)
}

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
```

<p align="center">
    <img src="https://raw.githubusercontent.com/emaldonadocruz/UTuning/master/figures/Catboost_Accuracy_plot.png"/>
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/emaldonadocruz/UTuning/master/figures/Catboost_example.png"/>
</p>


A second example using neural networks is coming soon.

<p align="center">
    <img src="https://raw.githubusercontent.com/emaldonadocruz/UTuning/master/figures/SearchSpace.png"/>
</p>

### Credits
-------
The dataset used for the examples is provided by Dr. Michael Pyrcz, GeostatsGuy: https://github.com/GeostatsGuy

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

Cookiecutter: https://github.com/audreyr/cookiecutter
`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


