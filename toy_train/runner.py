#!/usr/bin/env python

"""
Main Runner for the Toy Model Trainer.
"""

import json
import os

from trainer_utils import load_data_with_key
from training import run_toy_trainer
from constants import NEURAL_NET_MODEL_KEY, DEFAULT_NEURAL_NET_CONFIG_PATH, DEFAULT_TRAIN_SIZE, DEFAULT_SEED, IRIS_DATASET_KEY, KNN_MODEL_KEY

__author__ = "Steve DiCristofaro"


if __name__ == "__main__":

    # Command line interface
    import argparse
    parser = argparse.ArgumentParser(description='Train a model on a given dataset.')
    parser.add_argument('-data', dest='dataset', help='Module-based dataset to use.', type=str, default=IRIS_DATASET_KEY)
    parser.add_argument('-m', dest='model_key', help='Model to use.', type=str, default=KNN_MODEL_KEY)
    parser.add_argument('-tp', dest='train_percent', help='Percentage of data to allocate to training.', type=float, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument('--reg', dest='regressor', help='Perform regression.', default=False, action='store_true')
    parser.add_argument('--seed', dest='seed', help='Seed random number generators.', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # 1) Load a dataset
    data = None
    if args.dataset:
        data = load_data_with_key(args.dataset)
    else:
        raise RuntimeError('[!] No data source specified.')

    # 2) If training a deep learning network, load the structure from the JSON config
    nn_config = None
    if args.model_key == NEURAL_NET_MODEL_KEY:
        with open(os.path.join(os.path.dirname(__file__), DEFAULT_NEURAL_NET_CONFIG_PATH)) as config:
            nn_config = json.load(config)

    run_toy_trainer(data, args.model_key, train_size=args.train_percent,
                    random_seed=args.seed, regressor=args.regressor,
                    nn_config=nn_config)
