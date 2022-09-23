"""Main module."""

from UTuning import scorer
import numpy as np

from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import scipy.integrate as integrate

#sys.path.insert(0, r'C:\Users\eduar\OneDrive\PhD\UTuning')
#import sys
#sys.path.insert(0, r'C:\Users\em42363\OneDrive\PhD\UTuning')


# def GridSearchKeras(model, param_grid):

#     score = make_scorer(Goodness_loss_keras, greater_is_better=False)

#     random_cv = GridSearchCV(model,
#                             param_grid,
#                             cv = 2,
#                             #scoring='neg_mean_absolute_error',
#                             scoring=score,
#                             verbose = 2,
#                             n_jobs=1)
#     return random_cv

def RandomizedSearch(model, param_grid, cv=10, n_iter=10):

    score = make_scorer(Goodness_loss, greater_is_better=False)

    random_cv = RandomizedSearchCV(model,
                                   param_grid,
                                   cv=cv,
                                   n_iter=n_iter,
                                   n_jobs=-1,
                                   scoring=score,
                                   verbose = 2)
    return random_cv

def Grid(model, param_grid, cv):

    score = make_scorer(Goodness_loss, greater_is_better=True)

    random_cv = GridSearchCV(model,
                            param_grid,
                            scoring=score,
                            verbose = 2)
    return random_cv

def ensemble(model, X_s, batch_size,y_s):
    #Take n_samples to draw a distribution
    n_samples = 50
    mc_predictions = np.zeros((n_samples, y_s.shape[0]))
    for i in range(n_samples):
        #y_p = mc_model.predict(X_test, batch_size=4)
        y_p = model.predict(X_s, verbose=1, batch_size=batch_size)
        mc_predictions[i] = (y_p)
    return mc_predictions

def Goodness_loss(y_true, y_pred):

    n_quantiles = 11
    perc = np.linspace(0.0, 1.00, n_quantiles)
    Samples = 10

    Pred = y_pred[:, 0]
    Sigma = np.sqrt(y_pred[:, 1])
    Truth = y_true

    Pred_array = np.zeros((Sigma.shape[0], Samples))

    IF_array = np.zeros((Pred.shape[0], n_quantiles))

    for i in range(len(Pred)):
        Pred_array[i, :] = np.random.normal(loc=Pred[i], scale=Sigma[i], size=Samples)
        IF= scorer.APG_calc(Truth[i],
                            Pred_array[i, :],
                            Sigma[i],
                            n_quantiles)
        IF_array[i, :] = IF
    
    avgIndFunc = np.mean(IF_array, axis=0)
    
    a = np.zeros(len(avgIndFunc))
    for i in range(len(avgIndFunc)):
        if avgIndFunc[i] > perc[i] or avgIndFunc[i] == perc[i]:
            a[i] = 1
        else:
            a[i] = 0
    
    Sum = (3*a-2)*(avgIndFunc-perc)
    Goodness = 1-integrate.simps(Sum, perc)
    Accuracy = integrate.simps(a, perc)
    Prec = a*(avgIndFunc-perc)
    Precision = 1-2*integrate.simps(Prec, perc)
    d = y_true - y_pred[:,0]
    mae = np.mean(abs(d))
    #return (Accuracy+Goodness)
    return (0.95*(Accuracy) + 0.05*(1-Goodness))
    #return mae