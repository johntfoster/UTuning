import matplotlib.pyplot as plt
import numpy as np
import random

import matplotlib.tri as tri
from scipy.ndimage.filters import gaussian_filter

#import matplotlib
#matplotlib.rcParams['text.usetex'] = True


def error_line(Prediction, Truth, Sigma, Frac=1):
    '''Simple function to draw an error line plot. It takes three arrays of the same length

    Parameters
    ----------

    Prediction : float array
        The predicted value array (Prediction)
    Truth : float array
        The truth value array (Truth)
    Sigma : float array
        The standard deviation array (Sigma)
    Frac : float
        Frac is the fraction of points to display randomly

    Returns
    -------
    Scatter plot
    '''
    yline = [min(Prediction.min(), Truth.min()),
             max(Prediction.max(), Truth.max())]
    xline = [min(Prediction.min(), Truth.min()),
             max(Prediction.max(), Truth.max())]

    fig, ax = plt.subplots(figsize=(10, 6))
    # To display randomly less points [Remove , Keep] in fraction
    mask = np.random.choice([False, True], len(Prediction), p=[(1-Frac), (Frac)])
    ax.errorbar(Prediction[mask], Truth[mask], xerr=Sigma[mask],
                fmt='k.',
                ecolor='k')
    ax.plot(xline, yline, '-k')
    ax.set_xlabel('Predicted value, $\hat{y}$')
    ax.set_ylabel('True value, $y$ ')
    plt.show()

def goodness_plot(percentile,IF_array,Prediction_array,Truth,Sigma,minmax='True'):
    '''Simple function to draw an error line plot and its corresponding accuracy plot.

    Parameters
    ----------

    Prediction : float array
        The predicted value array (Prediction)
    Truth : float array
        The truth value array (Truth)
    Sigma : float array
        The standard deviation array (Sigma)
    Frac : float
        Frac is the fraction of points to display randomly

    Returns
    -------
    Scatter plot
    '''
    avgIndFunc = np.mean(IF_array, axis=0)

    L = 10
    mean = np.empty((L, len(percentile)))

    for p_interv in range(len(percentile)):
        for l in np.arange(0, L):
            samples = random.choices(IF_array[:, p_interv],
                                     k=IF_array.shape[0])
            mean[l, p_interv] = np.mean(samples)

    fig, ax =plt.subplots(figsize=(6,4))

    ax.plot(percentile, avgIndFunc,'-ok',markersize=5, label='Goodness')
    ax.plot(percentile,np.round(avgIndFunc+np.std(mean, axis=0), 3),'--k', label=None)
    ax.plot(percentile,np.round(avgIndFunc-np.std(mean, axis=0), 3),'--k', label='std')
    ax.plot([0, 1],[0, 1],'r--', label='1:1 line')
    ax.set_ylabel(r"$\overline{\xi (p)}$")
    ax.set_xlabel('Probability interval $p$')
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)


    return fig, ax


def error_accuracy_plot(percentile,IF_array,Prediction_array,Truth,Sigma,minmax='True'):
    '''Simple function to draw an error line plot and its corresponding accuracy plot.

    Parameters
    ----------

    Prediction : float array
        The predicted value array (Prediction)
    Truth : float array
        The truth value array (Truth)
    Sigma : float array
        The standard deviation array (Sigma)
    Frac : float
        Frac is the fraction of points to display randomly

    Returns
    -------
    Scatter plot
    '''
    avgIndFunc = np.mean(IF_array, axis=0)

    L = 10
    mean = np.empty((L, len(percentile)))

    for p_interv in range(len(percentile)):
        for l in np.arange(0, L):
            samples = random.choices(IF_array[:, p_interv],
                                     k=IF_array.shape[0])
            mean[l, p_interv] = np.mean(samples)

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))

    if len(Prediction_array.shape)>1:
        if minmax=='True':
            xline = [0,max(np.mean(Prediction_array,axis=1).max(),Truth.max())+max(np.mean(Prediction_array,axis=1).max(),Truth.max())*0.1]#
            yline = [0,xline[1]]#
        else:
            xline = [min(np.mean(Prediction_array,axis=1).min(),Truth.max()),max(np.mean(Prediction_array,axis=1).max(),Truth.max())+max(np.mean(Prediction_array,axis=1).max(),Truth.max())*0.1]#
            yline = [min(np.mean(Prediction_array,axis=1).min(),Truth.min()),xline[1]]#
        ax1.errorbar(np.mean(Prediction_array,axis=1), Truth, xerr=Sigma,
                     fmt='k.',
                     ecolor='k')
    else:
        if minmax=='True':
            xline = [0,max(Prediction_array.max(),Truth.max())+max(Prediction_array.max(),Truth.max())*0.1]#
            yline = [0,xline[1]]#
        else:
            xline = [min(np.mean(Prediction_array,axis=1).min(),Truth.max()),max(np.mean(Prediction_array,axis=1).max(),Truth.max())+max(np.mean(Prediction_array,axis=1).max(),Truth.max())*0.1]#
            yline = [min(np.mean(Prediction_array,axis=1).min(),Truth.min()),xline[1]]#


        ax1.errorbar(Prediction_array, Truth, xerr=Sigma,
                     fmt='k.',
                     ecolor='k')
    ax1.plot(xline, yline, '-k')
    ax1.set_xlabel('Predicted value, $\hat{y}$')
    ax1.set_ylabel('True value, $y$ ')

    ax2.plot(percentile, avgIndFunc,'-ok',markersize=5)
    ax2.plot(percentile,np.round(avgIndFunc+np.std(mean, axis=0), 3),'--k')
    ax2.plot(percentile,np.round(avgIndFunc-np.std(mean, axis=0), 3),'--k')
    ax2.plot([0, 1],[0, 1],'-k')
    ax2.set_ylabel(r"$\overline{\xi (p)}$")
    ax2.set_xlabel('Probability interval $p$')
    ax2.set_ylim(0,1)
    ax2.set_xlim(0,1)

    ax2.plot(percentile, avgIndFunc,'-ok',markersize=5)


def surface(x, y, z, levels, labels):

    fig, (ax1) = plt.subplots(nrows=1,figsize=(12,6))

    npoints=x.shape[0]
    smooth=1

    # Create grid values first.
    xi = np.linspace(x.min(), x.max(), npoints)
    yi = np.linspace(y.min(), y.max(), npoints)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    zi = gaussian_filter(zi, smooth)

    levels = levels

    ax1.contour(xi, yi, zi, levels=levels, linewidths=0.1, colors='k')

    cntr1 = ax1.contourf(xi, yi, zi, levels=levels, cmap="inferno",alpha=0.95)

    cbar = plt.colorbar(cntr1, ax=ax1)
    cbar.set_label(labels['z'], rotation=270,labelpad=30)

    ax1.set(xlim=(x.min(), x.max()),ylim=(y.min(), y.max()))

    ax1.scatter(x,y,s=7,color='white')
    ax1.set_xlabel(labels['x'])
    ax1.set_ylabel(labels['y'])


    plt.show()
