=================================
Hyperparameter Uncertainty Tuning
=================================


.. image:: https://img.shields.io/pypi/v/UTuning.svg
        :target: https://pypi.python.org/pypi/UTuning

.. image:: https://img.shields.io/travis/emaldonadocruz/UTuning.svg
        :target: https://travis-ci.com/emaldonadocruz/UTuning

.. image:: https://readthedocs.org/projects/UTuning/badge/?version=latest
        :target: https://UTuning.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/emaldonadocruz/UTuning/shield.svg
     :target: https://pyup.io/repos/github/emaldonadocruz/UTuning/
     :alt: Updates



Uncertainty Tuning (UTuning) is a package that focuses on summarizing uncertainty model performance for optimum hyperparameter tuning.

.. image:: https://raw.githubusercontent.com/emaldonadocruz/UTuning/master/figures/CrossVal.png

Comparison of the cross-validation plot and respective accuracy plot for two uncertainty models where the hyperparameters were optimized using different objective functions. a) Using MAE, b) Uncertainty model goodness.

* Free software: BSD license
* Documentation: https://UTuning.readthedocs.io.

Description
-----------
* Current machine learning models focus on prediction accuracy and minimizing prediction error. However, when uncertainty is present, predicting a single estimate must be replaced with a prediction of the uncertainty distribution.
* We propose UTuning, a python-based library to evaluate and tune machine learning models for maximum model goodnes, accuracy and precision.
* UTuning enables the use of machine learning for any spatial, sparsely sampled settings, including energy geothermal, hydrocarbons, solar, wind and other problems dealing with uncertainty.

Features
--------

* Hyperparameter tuning for ensemble based uncertainty models
* Scikit wrapper for grid search and random search

Usage
-----


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
