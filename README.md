# toy_trainer
A toy Python package to train sklearn and Tensorflow models.

## How to Run
```python
# Most basic (will use all defaults)
python runner.py

# More specific (classification)
python runner.py -data DIGITS -m SVM -tp 0.6

# More specific (regression)
python runner.py -data BOS -m SVM -tp 0.6 --reg

```

#### Configurable Parameters
- Dataset [-data] The key to use to load a given dataset.
  - Supported keys:
    - IRIS [Iris Classification Dataset](https://archive.ics.uci.edu/ml/datasets/Iris)
    - BOS [Boston Housing Regression Dataset](https://archive.ics.uci.edu/ml/datasets/Housing)
    - DIGITS [Handwritten Digits Classification Dataset](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits)
- Model Key [-m] The key to use to specify the model to train.
  - Supported keys:
    - KNN (K-Nearest Neighbors)
    - SVM (Support Vector Machine)
    - RF (Random Forest)
    - NEURAL (Tensorflow Neural Network)
   NOTE:
   Optionally enable regression [`--`reg] with any model key except NEURAL (only classification is supported currently)
 - Training Percent [-tp] A float between 0.0 and 1.0 to designate the proportion of data to use for training.
 - Random seed [`--`seed] An integer seed for the random number generator.

#### Tensorflow Configuration
The configuration of deep learning networks resides in nn_config.json. The configuration file supports very basic layers involving dropout regularizers and relevant supported Tensorflow activation functions. This configuration requires a priori knowledge of your dataset to properly configure the weights for each layer using the n_weights parameter. The following configurable keys are supported:
  - Optimizer Type
    - rmsprop [RMSPropOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer)
    - gradient_descent [GradientDescentOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer)
  - Optimizer Arguments
    - Any keyword arguments pertaining to the optimizer object can be specified
  - Layer Type
    - dropout [tf.nn.dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout)
    NOTE: There is also a configurable dropout mapping for the probabilities of the input and hidden layer dropouts that accepts floating-point probabilities
  - Layer activation [Activation Functions](https://www.tensorflow.org/api_guides/python/nn#Activation_Functions)
    - sigmoid, tanh, (Untested: elu, softplus, and softsign)
    - relu, (Untested: relu6, crelu and relu_x)

#### Some Known Issues
- Multiprocessing crash if trying to use classification on Boston Dataset

#### Some Future Additions
- Distutils formatting
- Deeper testing of Tensorflow functionality
- Loading data from csv
- Generic cross validation
- Model saving and loading (with dry run flag)
- More robust error checking on configurations
- Direct specification of model parameters (sklearn) via CLI
- Lots of unit tests
- More grid search types
- Tensorflow regression
- Module loading based on configurations
- More sklearn model objects and Tensorflow layers
- More performance metrics
