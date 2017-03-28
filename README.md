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
    - NEURAL
   NOTE:
   Optionally enable regression [`--`reg] with any model key except NEURAL (only classification is supported currently)
 - Training Percent [-tp] A float between 0.0 and 1.0 to designate the proportion of data to use for training.
 - Random seed [`--`seed] An integer seed for the random number generator.

#### Some Known Issues
- Multiprocessing crash if trying to use classification on Boston Dataset

#### Some Future Additions
- Distutils formatting
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
