#!/usr/bin/env python
"""
Functions for training Tensorflow networks and sklearn models.
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer

from trainer_utils import profile, ConfigurationError, load_optimizer_by_type, load_regularizer_by_type, load_activation_by_type
from trainer_utils import load_model_by_key, load_default_grid_by_key
from constants import DEFAULT_BATCH_SIZE, DEFAULT_NETWORK_ITER

__author__ = "Steve DiCristofaro"

##############
# TENSORFLOW #
##############

def model(layers, weights, X, p_keep_input, p_keep_hidden):
    '''
    Build the neural network structure.
    :param layers: Ordered list of layer configurations
    :param weights: Ordered list of weights per layer
    :param X: Input features
    :param p_keep_input: Input dropout probability
    :param p_keep_hidden: Hidden layer dropout probability
    :return: Tensorflow network
    '''
    if len(layers) == 0:
        return tf.matmul(X, weights[0])
    o = None
    for w, l in zip(weights, layers):
        type = load_regularizer_by_type(l['type'])
        activation = load_activation_by_type(l.get('activation'))
        if o is None:
            o = activation(tf.matmul(type(X, p_keep_input), w))
        elif activation is None:
            o = tf.matmul(type(o, p_keep_hidden), w)
        else:
            o = activation(tf.matmul(type(o, p_keep_hidden), w))
    return o

def init_weights(shape, stddev=0.01):
    '''
    Initialize a random set of weights of a given dimension.
    :param shape: Length 2 list of [n, m] dimensions
    :return: tf.Variable of random weights
    '''
    return tf.Variable(tf.random_normal(shape, stddev=stddev))

def assemble_weight_list(layers, m, k):
    '''
    Gather the ordered list of weights for a given network.
    :param layers: Ordered list of layer configurations
    :param m: Feature dimension
    :param k: Class label dimension
    :return: Ordered list of weights
    '''
    weights = []
    if not len(layers):
        weights.append(init_weights([m, k]))
    else:
        for idx, layer in enumerate(layers):
            if idx == 0:
                # The first layer weights are dimension [number of features X specified weights]
                weights.append(init_weights([m, layer['n_weights']]))
            else:
                prev_layer_w = layers[idx - 1]['n_weights']
                if idx == len(layers) - 1:
                    # The last layer weights are dimension [previous layer weights X number of classes]
                    weights.append(init_weights([prev_layer_w, k]))
                else:
                    # Intermediate layer weights are dimension [previous layer weights X specified weights]
                    weights.append(init_weights([prev_layer_w, layer['n_weights']]))
    return weights

def train_tensorflow_network(config, train_features, test_features, train_labels, test_labels):
    '''
    Train a tensorflow network on a dataset.
    :param config: Serialized network config
    :param train_features: np.array of features
    :param test_features: np.array of features
    :param train_labels: array of class labels
    :param test_labels: array of class labels
    '''

    # Gather data dimensions to use for placeholder parameters
    n, m = train_features.shape
    _, k = train_labels.shape

    # Define placeholder variables
    X = tf.placeholder("float", [None, m])
    Y = tf.placeholder("float", [None, k])

    # NOTE: Empty layers defaults to a single matmul operation
    layers = config.get('layers', [])

    # Assemble weights for network layers
    weights = assemble_weight_list(layers, m, k)

    # Define placeholders for dropout layer probabilities
    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    # Build network structure
    network = model(layers, weights, X, p_keep_input, p_keep_hidden)

    # Define cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=Y))

    # Get optimizer parameters
    optimizer = config.get('optimizer')
    if not optimizer:
        raise ConfigurationError('[!] No optimizer parameters specified in network configuration!')
    optimizer_args = {k: v for (k, v) in optimizer.iteritems() if k != 'type'}

    # Define training optimizer
    train_op = load_optimizer_by_type(optimizer.get('type'))(**optimizer_args).minimize(cost)

    # Define evaluation function
    predict_op = tf.argmax(network, 1)


    # Grab optional parameters
    batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE)
    n_iter = config.get('n_iter', DEFAULT_NETWORK_ITER)

    # Grab dropout parameters
    train_p_input = config.get('dropout').get('train_p_input')
    train_p_hidden = config.get('dropout').get('train_p_hidden')
    test_p_input = config.get('dropout').get('test_p_input')
    test_p_hidden = config.get('dropout').get('test_p_hidden')

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()

        for i in range(n_iter):
            for start, end in zip(range(0, n, batch_size), range(batch_size, n + 1, batch_size)):
                sess.run(train_op, feed_dict={X: train_features[start:end], Y: train_labels[start:end],
                                              p_keep_input: train_p_input, p_keep_hidden: train_p_hidden})

                print '[i] EPOCH[%s] -- SCORE[%s]' % (i, np.mean(np.argmax(test_labels, axis=1) == sess.run(predict_op,
                                                                            feed_dict={X: test_features,
                                                                                       p_keep_input: test_p_input,
                                                                                       p_keep_hidden: test_p_hidden})))

############
# SK-Learn #
############

def train_sklearn_model(train_features, test_features, train_labels, test_labels, model_key, regressor=False):
    '''
    Train sklearn model on a given dataset.
    :param train_features: np.array of features
    :param test_features: np.array of features
    :param train_labels: array of class labels
    :param test_labels: array of class labels
    :param model_key: Model key string
    :param regressor: Regressive flag
    '''
    # Load a model object and associated parameters
    model = load_model_by_key(model_key, regressor=regressor)
    param_grid = load_default_grid_by_key(model_key)

    # Tune hyperparameters
    grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, verbose=True)
    clf = fit_model(grid_search, train_features, train_labels)

    # Grab the highest performing parameter set
    best_params = {param: getattr(clf.best_estimator_, param) for param in param_grid}

    # Train final model
    clf_final = load_model_by_key(model_key, regressor=regressor, **best_params)
    clf_final = fit_model(clf_final, train_features, train_labels)
    score_final = clf_final.score(test_features, test_labels)

    if regressor:
        print '[i] R2 Score: %s' % score_final
    else:
        print '[i] Accuracy Score: %s' % score_final

##################
# Main Interface #
##################

def run_toy_trainer(dataset, model_key, train_size, random_seed, regressor, nn_config):
    '''
    Main interface for model training.
    :param dataset: Dataset object
    :param model_key: Model key string
    :param train_size: Percent of data to allocate for training
    :param random_seed: Random number generator seed
    :param regressor: Regressive flag
    :param nn_config: Serialized network configuration
    '''

    # 1) Partition the dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=train_size, random_state=random_seed)

    if nn_config:
        # Training branch for Tensorflow networks
        if not regressor:
            # Training a neural network classifier requires one-of-k encoding
            lb = unary_2_one_of_k(y_train)
            y_train = lb(y_train)
            y_test = lb(y_test)
        else:
            raise ConfigurationError('[!] Unable to train regressive Tensorflow model at the moment.')

        train_tensorflow_network(nn_config, X_train, X_test, y_train, y_test)

    else:
        # Training branch for sklearn models
        train_sklearn_model(X_train, X_test, y_train, y_test, model_key, regressor=regressor)


##################
# Helper Functions
##################
@profile
def fit_model(model, features, labels):
    '''
    Profiled function for fitting a sklearn model object to a feature set.
    :param model: sklearn model object
    :param features: np.array of features
    :param labels: np.array or list of class labels
    :return: trained sklearn model
    '''
    print '[i] Fitting %s object' % model.__class__.__name__
    return model.fit(features, labels)

def unary_2_one_of_k(labels):
    '''
    Convert a list of categorical variables into one-of-k format.
    :param labels: np.array or list of unary class labels
    :return: transform function for one-of-k encoding
    '''
    return LabelBinarizer().fit(labels).transform