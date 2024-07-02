"""A module for creating and training machine learning models for landslide prediction.

This module provides a class called `model` which allows users to create and train
machine learning models for landslide prediction. The module supports two types of 
models: RandomForest and SVM. Users can specify the model type, filepath to the dataset, 
test size for train-test split, and other optional parameters.

"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

class MlModel:
    """
    A class for creating and training machine learning models for landslide prediction.
    """

    def __init__(self,
                  filepath=None,
                  model_type='RandomForest',
                  test_size=0.2,
                  features=None,
                  **kwargs):
        valid_model_types = ['RandomForest', 'SVM', 'GBM']
        if model_type not in valid_model_types:
            raise ValueError('Model type not supported.')
        if test_size <= 0 or test_size >= 1:
            raise ValueError('Test size must be between 0 and 1.')
        if filepath is not None:
            if not isinstance(filepath, str):
                raise TypeError('Filepath must be a string.')
        if not isinstance(features, list):
            raise TypeError('Features must be a list.')
        if not all(isinstance(feature, str) for feature in features):
            raise TypeError('Features must be a list of strings.')
        #TODO: check if features are in the dataset
        self.filepath = filepath
        self.test_size = test_size
        self.kwargs = kwargs
        self.features = features
        self.model = self._initialize_model(model_type)
        self._load_data()
        self._preprocess_data(features, 'class')
        self.model.fit(self.x_train, self.y_train)
        self.accuracy, self.report = self._evaluate_model()

    def _initialize_model(self, model_type):
        if model_type == 'RandomForest':
            return RandomForestClassifier(**self.kwargs)
        elif model_type == 'SVM':
            return SVC(**self.kwargs)
        elif model_type == 'GBM':
            return GradientBoostingClassifier(**self.kwargs)
        else:
            raise ValueError('Model type not supported.')

    def _load_data(self):
        """
        Load the data from the specified filepath.
        """
        self.data = pd.read_csv(self.filepath)
        return self.data

    def _preprocess_data(self, feature_columns, target_column):
        """
        Preprocess the data by splitting it into training and testing sets.
        """
        x = self.data[feature_columns]
        y = self.data[target_column]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=42)

    def _evaluate_model(self):
        """
        Evaluate the model on the test data.
        """
        y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        return accuracy, report

    def predict(self, filepath):
        """
        Run the model for another dataset.
        """
        self.model.predict(filepath)