"""A module for creating and training machine learning models for landslide prediction.

This module provides a class called `model` which allows users to create and train
machine learning models for landslide prediction. The module supports two types of 
models: RandomForest and SVM. Users can specify the model type, filepath to the dataset, 
test size for train-test split, and other optional parameters.

"""

import os
import warnings
import joblib
from netCDF4 import Dataset #pylint: disable=no-name-in-module
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from .config import VALID_MODELS

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
        self.filepath = filepath
        self.model_type = model_type
        self.target_column = target_column
        self.features_list = features_list
        self.test_size = test_size
        self.kwargs = kwargs['kwargs']
        # Verify the input arguments
        self.__verify_input()
        # Load and preprocess the dataset
        self.__load_dataset()
        self.__preprocess_data()
        self.model = self.__initialize_model()

        # Initialize the model attributes
        self.y_pred = None
        self.y_pred_test = None
        self.report = None
        self.prediction_data_path = None
        self.prediction_file_path = None
        self.last_prediction = None

    def __initialize_model(self):
        """
        Initialize the machine learning model based on the specified model type.
        """
        if self.model_type == 'RandomForest':
            return RandomForestClassifier(**self.kwargs)
        elif self.model_type == 'SVM':
            return SVC(**self.kwargs)
        elif self.model_type == 'GBM':
            return GradientBoostingClassifier(**self.kwargs)
        else:
            raise ValueError('Model type not supported.')

    def __load_dataset(self):
        """
        Load the data from the specified filepath.
        """
        self.dataset = pd.read_csv(self.filepath, header=0)

    def __preprocess_data(self):
        """
        Preprocess the data by splitting it into training and testing sets.
        """
        x = self.dataset[self.features_list]
        y = self.dataset[self.target_column]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=42)

    def __verify_input(self):
        """
        Verify the input arguments for the model.
        """
        if self.model_type not in VALID_MODELS:
            raise ValueError('Model type not supported.')
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(f"File '{self.filepath}' does not exist.")
        if not isinstance(self.target_column, str):
            raise TypeError('Target must be a string.')
        if not isinstance(self.features_list, list):
            raise TypeError('Features must be a list.')
        if not all(isinstance(feature, str) for feature in self.features_list):
            raise TypeError('Features must be a list of strings.')
        if not isinstance(self.test_size, float):
            raise TypeError('Test size must be a float.')
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError('Test size must be between 0 and 1.')

    def setup(self, **kwargs):
        """
        Reconfigure the model with new parameters.

        Args:
            **kwargs: Keyword arguments to be passed to the machine learning model.
        """
        # Check if the given kwargs are within the allowed kwargs for the model
        invalid_kwargs = [kwarg for kwarg in kwargs if kwarg not in self.model.get_params()]
        if invalid_kwargs:
            raise ValueError(f"Invalid kwargs found: {', '.join(invalid_kwargs)}")
        # Update the kwargs with the new parameters
        self.kwargs.update(kwargs)
        # Reinitialize the model with the updated kwargs
        self.model = self.__initialize_model()
        self.model.fit(self.x_train, self.y_train)
        self.y_pred_test = self.model.predict(self.x_test)

    def evaluate_model(self, *, show:bool=False):
        """
        Evaluate the performance of the trained model.
        """
        if not isinstance(show, bool):
            raise TypeError('Plot must be a boolean.')
        if self.y_pred is None:
            warnings.warn("No data was loaded. Prediction will be done with test data.")
            self.y_pred = self.model.predict(self.x_test)
        if show is True:
            print(classification_report(self.y_test, self.y_pred, output_dict=False))
        self.report = classification_report(self.y_test, self.y_pred, output_dict=True)
        return self.report

    def predict(self, data):
        """
        Make predictions using the trained model.

        Args:
            x (array-like): The input features for making predictions.
            It can be a pandas DataFrame, an xarray Dataset, or a file path to a CSV or NetCDF file.

        Returns:
            array: The predicted values.
        """
        type_of_data = type(data)
        if type_of_data == pd.DataFrame:
            data_to_predict = data[self.features_list]
        elif type_of_data == xr.Dataset:
            data_to_predict = data.to_dataframe()[self.features_list]
        elif type_of_data == str:
            if not os.path.isfile(data):
                raise FileNotFoundError(f"File '{data}' does not exist.")
            elif data.endswith('.csv'):
                data_to_predict = pd.read_csv(data, header=0)[self.features_list]
            elif data.endswith('.nc'):
                ds = Dataset(data) 
                print(ds)
                data_to_predict = ds.to_dataframe()[self.features_list]


        self.last_prediction = self.model.predict(data_to_predict)
        return self.last_prediction

    def save_model(self, filepath):
        """
        Save the trained model to a file.

        Args:
            filepath (str): The filepath to save the model to.
        """
        if not isinstance(filepath, str):
            raise ValueError('Filepath must be a string.')
        joblib.dump(self, filepath)

    def generate_heatmap(self, filepath):
        """
        Save the heatmap of the dataset to a file.

        Args:
            filepath (str): The filepath to save the heatmap to.
        """
        if not isinstance(filepath, str):
            raise TypeError('Filepath must be a string.')
        plt.figure(figsize=(10, 8))
        numeric_data = self.dataset.select_dtypes(include=[float, int])
        if self.target_column in numeric_data.columns:
            numeric_data = numeric_data.drop(columns=[self.target_column])
        keywords = ['coord', 'loc', 'location', 'coordinates']
        columns_to_exclude = [col for col in numeric_data.columns
                              if any(keyword in col.lower() for keyword in keywords)]
        numeric_data = numeric_data.drop(columns=columns_to_exclude)
        sns.heatmap(numeric_data.corr(),
                    xticklabels=numeric_data.columns,
                    yticklabels=numeric_data.columns,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm')
        plt.title('Heatmap of Dataset Features')
        plt.savefig(filepath)
