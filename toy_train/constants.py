#!/usr/bin/env python
"""
Heap of constants used in the training of models.
"""
__author__ = "Steve DiCristofaro"

# Module paths
PYTHON_TENSORFLOW_PATH = 'tensorflow.python.'
CONTINUOUS_ACTIVATION_MODULE = PYTHON_TENSORFLOW_PATH + 'ops.gen_nn_ops'
NONLINEAR_ACTIVATION_MODULE = PYTHON_TENSORFLOW_PATH + 'ops.math_ops'
REGULARIZER_ACTIVATION_MODULE = PYTHON_TENSORFLOW_PATH + 'ops.nn_ops'
RMSPROP_OPTIMIZER_MODULE = PYTHON_TENSORFLOW_PATH + 'training.rmsprop'
GRADIENT_OPTIMIZER_MODULE = PYTHON_TENSORFLOW_PATH + 'training.gradient_descent'

# Neural network configuration file
DEFAULT_NEURAL_NET_CONFIG_PATH = 'nn_config.json'

# Training defaults
DEFAULT_TRAIN_SIZE = 0.7
DEFAULT_VALIDATION_SIZE = 0.1
DEFAULT_SEED = 12345
DEFAULT_BATCH_SIZE = 100
DEFAULT_NETWORK_ITER = 1000

# Dataset Keys
IRIS_DATASET_KEY = 'IRIS'
BOSTON_DATASET_KEY = 'BOS'
DIGITS_DATASET_KEY = 'DIGITS'
VALID_DATASET_KEYS = [IRIS_DATASET_KEY, BOSTON_DATASET_KEY, DIGITS_DATASET_KEY]

# Model Keys
KNN_MODEL_KEY = 'KNN'
SVM_MODEL_KEY = 'SVM'
RF_MODEL_KEY = 'RF'
NEURAL_NET_MODEL_KEY = 'NEURAL'
VALID_MODEL_KEYS = [KNN_MODEL_KEY, SVM_MODEL_KEY, RF_MODEL_KEY, NEURAL_NET_MODEL_KEY]

# Tensorflow layers
CONTINUOUS_ACTIVATIONS = ['relu', 'relu6', 'crelu', 'relu_x']
NONLINEAR_ACTIVATIONS = ['sigmoid', 'tanh', 'elu', 'softplus', 'softsign']
REGLARIZER_ACTIVATIONS = ['dropout']

# Optimizer parameters
GRADIENT_OPT_TYPE = 'gradient_descent'
GRADIENT_OPTIMIZER = 'GradientDescentOptimizer'
RMSPROP_OPTIMIZER = 'RMSPropOptimizer'

# Model Hyperparameter Defaults
DEFAULT_KNN_GRID = {
    "n_neighbors": [1, 5, 10, 15, 20]
}
DEFAULT_SVM_GRID = {
    "kernel": ["linear", "rbf"],
    "C": [0.001, 0.01, 0.1],
    "gamma": [0.5, 1, 2]
}
DEFAULT_RF_GRID = {
    "n_estimators": [1, 5, 10, 15, 20]
}
