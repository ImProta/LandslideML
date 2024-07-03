"""A module for creating and training machine learning models for landslide prediction.

This module provides a class called `model` which allows users to create and train
machine learning models for landslide prediction. The module supports two types of 
models: RandomForest and SVM. Users can specify the model type, filepath to the dataset, 
test size for train-test split, and other optional parameters.

"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from landslideml import VALID_MODELS
# from sklearn.metrics import classification_report, accuracy_score

class MlModel:
    """
    A class for creating and training machine learning models for landslide prediction.

    Attributes:
        model_type (str): The type of machine learning model to be used. Supported model types are 
            'RandomForest', 'SVM', and 'GBM'.
        filepath (str): The filepath of the dataset to be used for training and testing the model.
        target (str): The target variable in the dataset.
        features (list): The list of feature variables in the dataset.
        test_size (float): The proportion of the dataset to be used for testing the model.
        kwargs (dict): Additional keyword arguments to be passed to the machine learning model.
        type (str): The type of machine learning model.
        model: The initialized machine learning model.
        dataset: The loaded dataset.
        x_train: The training set features.
        x_test: The testing set features.
        y_train: The training set target variable.
        y_test: The testing set target variable.

    Args:
        model_type (str): The type of machine learning model to be used. Supported model types are 
            'RandomForest', 'SVM', and 'GBM'.
        filepath (str): The filepath of the dataset to be used for training and testing the model.
        target (str): The target variable in the dataset.
        features (list): The list of feature variables in the dataset.
        test_size (float): The proportion of the dataset to be used for testing the model.
        **kwargs: Additional keyword arguments to be passed to the machine learning model.

    Raises:
        ValueError: If the model type is not supported.
        TypeError: If the filepath is not a string, the target is not a string, the features
            are not a list, or the features are not strings.
    """

    def __init__(self,
                  filepath=None,
                  model_type='RandomForest',
                  target_column='label',
                  features_list=None,
                  test_size=0.2,
                  **kwargs):
        self.__verify_input(model_type, filepath, target_column, features_list, test_size)
        self.filepath = filepath
        self.type = model_type
        self.target_column = target_column
        self.features_list = features_list
        self.test_size = test_size
        self.kwargs = kwargs

        # Load and preprocess the dataset
        self._load_dataset()
        self._preprocess_data()

    def _initialize_model(self):
        match self.type:
            case 'RandomForest':
                return RandomForestClassifier()
            case 'SVM':
                return SVC()
            case 'GBM':
                return GradientBoostingClassifier()
            case _:
                raise ValueError('Model type not supported.')

    def _load_dataset(self):
        """
        Load the data from the specified filepath.
        """
        self.dataset = pd.read_csv(self.filepath, header=0)

    def _preprocess_data(self):
        """
        Preprocess the data by splitting it into training and testing sets.
        """
        x = self.dataset[self.features_list]
        y = self.dataset[self.target_column]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=42)

    def __verify_input(self, model_type, filepath, target, features, test_size):
        if model_type not in VALID_MODELS:
            raise ValueError('Model type not supported.')
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File '{filepath}' does not exist.")
        if not isinstance(target, str):
            raise TypeError('Target must be a string.')
        if not isinstance(features, list):
            raise TypeError('Features must be a list.')
        if not all(isinstance(feature, str) for feature in features):
            raise TypeError('Features must be a list of strings.')
        if not isinstance(test_size, float):
            raise TypeError('Test size must be a float.')
        if test_size <= 0 or test_size >= 1:
            raise ValueError('Test size must be between 0 and 1.')
