#!/usr/bin/env python
"""
Utility functions used in the training of models.
"""
from time import time
from importlib import import_module

from sklearn.datasets import load_iris, load_boston, load_digits
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from constants import IRIS_DATASET_KEY, BOSTON_DATASET_KEY, DIGITS_DATASET_KEY, VALID_DATASET_KEYS
from constants import KNN_MODEL_KEY, SVM_MODEL_KEY, RF_MODEL_KEY
from constants import DEFAULT_KNN_GRID, DEFAULT_SVM_GRID, DEFAULT_RF_GRID
from constants import REGULARIZER_ACTIVATION_MODULE, NONLINEAR_ACTIVATION_MODULE, CONTINUOUS_ACTIVATION_MODULE
from constants import GRADIENT_OPTIMIZER_MODULE, RMSPROP_OPTIMIZER_MODULE
from constants import CONTINUOUS_ACTIVATIONS, REGLARIZER_ACTIVATIONS, GRADIENT_OPT_TYPE, GRADIENT_OPTIMIZER, RMSPROP_OPTIMIZER

__author__ = "Steve DiCristofaro"

def profile(function):
    '''
    Wrapper function for displaying execution time.
    :param function: A function object
    :return: function wrapper object
    '''
    def wrap(*args, **kwargs):
        start = time()
        result = function(*args, **kwargs)
        duration = time() - start
        print '[i] Executed %s in %s seconds' % (function.__name__, duration)
        return result
    return wrap

def load_data_with_key(key):
    '''
    Load a dataset from a module based on a key.
    :param key: string dataset key
    :return: Dataset object
    '''
    @profile
    def _load_data(key):
        if key == IRIS_DATASET_KEY:
            return load_iris()
        elif key == BOSTON_DATASET_KEY:
            return load_boston()
        elif key == DIGITS_DATASET_KEY:
            return load_digits()

    if key in VALID_DATASET_KEYS:
        return _load_data(key)

    raise ConfigurationError('[!] Invalid dataset key specified [%s].' % key)

def load_model_by_key(key, regressor=False):
    '''
    Load an sklearn model by key.
    :param key: Model key string
    :param regressor: Regression flag
    :return: sklearn model object
    '''
    if key == KNN_MODEL_KEY:
        return KNeighborsRegressor() if regressor else KNeighborsClassifier()
    elif key == SVM_MODEL_KEY:
        return SVR() if regressor else SVC()
    elif key == RF_MODEL_KEY:
        return RandomForestRegressor() if regressor else RandomForestClassifier()
    raise ConfigurationError('[!] Invalid model key specified [%s]' % key)

def load_default_grid_by_key(key):
    '''
    Load default map of grid parameters for hyperparameter tuning.
    :param key: Model key string
    :return: dict of parameters
    '''
    if key == KNN_MODEL_KEY:
        return DEFAULT_KNN_GRID
    elif key == SVM_MODEL_KEY:
        return DEFAULT_SVM_GRID
    elif key == RF_MODEL_KEY:
        return DEFAULT_RF_GRID
    raise ConfigurationError('[!] Invalid model key specified [%s]' % key)

def load_regularizer_by_type(type):
    '''
    Load Tensorflow regularization layer.
    :param type: Regularizer type
    :return: Regularizer function
    '''
    if not type:
        return
    if type in REGLARIZER_ACTIVATIONS:
        return getattr(import_module(REGULARIZER_ACTIVATION_MODULE), type)

def load_activation_by_type(type):
    '''
    Load an activation function.
    :param type: Name string of activation function
    :return: The activation function
    '''
    if not type:
        return
    if type in CONTINUOUS_ACTIVATIONS:
        return getattr(import_module(CONTINUOUS_ACTIVATION_MODULE), type)
    return getattr(import_module(NONLINEAR_ACTIVATION_MODULE), type)

def load_optimizer_by_type(type):
    '''
    Load an optimizer object by type.
    :param type: Optimizer type string
    :return: Optimizer object
    '''
    if not type:
        raise ConfigurationError('[!] Could not find type for optimizer!')
    if type == GRADIENT_OPT_TYPE:
        return getattr(import_module(GRADIENT_OPTIMIZER_MODULE), GRADIENT_OPTIMIZER)
    return getattr(import_module(RMSPROP_OPTIMIZER_MODULE), RMSPROP_OPTIMIZER)

class ConfigurationError(Exception):
    '''
    Custom exception to be thrown on corrupt configurations.
    '''
    pass