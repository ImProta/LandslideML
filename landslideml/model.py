"""
A module for creating and training machine learning models for landslide prediction.

This module provides a class called `MlModel` which allows users to create and train
machine learning models for landslide prediction. The module supports three types of 
models: RandomForest, SVM, and Gradient Boosting (GBM). Users can specify the model type,
filepath to the dataset, test size for train-test split, and other optional parameters.
"""

import os
import warnings
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from .config import VALID_MODELS

class MlModel:
    """
    A class for creating and training machine learning models for landslide prediction.

    Input:
        filepath (str): The filepath of the dataset to be used for training and testing the model.
        model_type (str): The type of machine learning model to be used. Supported model types are 
            'RandomForest', 'SVM', and 'GBM'.
        target_column (str): The target variable in the dataset.
        features_list (list): The list of feature variables in the dataset.
        test_size (float): The proportion of the dataset to be used for testing the model.
        **kwargs: Additional keyword arguments to be passed to the machine learning model.

    Attributes:
        filepath (str): The filepath of the dataset to be used for training and testing the model.
        model_type (str): The type of machine learning model to be used. Supported model types are 
            'RandomForest', 'SVM', and 'GBM'.
        target_column (str): The target variable in the dataset.
        features_list (list): The list of feature variables in the dataset.
        test_size (float): The proportion of the dataset to be used for testing the model.
        kwargs (dict): Additional keyword arguments to be passed to the machine learning model.
        model: The initialized machine learning model.
        dataset(pd.Dataframe): The loaded dataset from the specified filepath.
        x_train(pd.Dataframe): The training set features.
        x_test(pd.Dataframe): The testing set features.
        y_train(pd.Dataframe): The training set target variable.
        y_test(pd.Dataframe): The testing set target variable.
        report(dict): The classification report of the trained model.
        prediction_dataset_size(int): The size of the prediction dataset.
        prediction_location(pd.Dataframe): The location columns of the prediction dataset.
        prediction_object(object): The input data for making predictions.
        prediction_object_type(type): The type of the input data for making predictions.
        prediction(array): The predicted values.
        prediction_map(pd.Dataframe): The mapped prediction values to the original location in dataset.

    Methods:
        __init__: Initializes the MlModel class with the specified parameters.
        __initialize_model: Initializes the machine learning model of the specified model type.
        __load_dataset: Loads the data from the specified filepath.
        __mapping: Maps the prediction values to the original entries in the prediction dataset.
        __preprocess_data: Preprocesses the data by splitting it into training and testing sets.
        __verify_input: Verifies the input arguments for the model.
        setup: Reconfigures the model setup with new parameters.
        evaluate_model: Evaluates the performance of the trained model.
        predict: Makes predictions using the current trained model.
        save_model: Saves the trained model to a file.
        generate_heatmap: Saves the heatmap of the dataset to a file.    

    Raises:
        ValueError: If the model type is not supported.
        FileNotFoundError: If the filepath does not exist.
        TypeError: If the target_column is not a string, the features_list
            is not a list, or the features_list contains non-string elements.
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
        self.__verify_input()
        self.__load_dataset()
        self.__preprocess_data()
        self.model = self.__initialize_model()
        self.y_pred = None
        self.y_pred_test = None
        self.report = None
        self.prediction_dataset_size = None
        self.prediction_location = None
        self.prediction_object = None
        self.prediction_object_type = None
        self.prediction = None
        self.prediction_map = None

    def __initialize_model(self):
        """
        Initialize the machine learning model based on the specified model type.

        Raises:
            ValueError: If the model type is not supported.
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

    def __mapping(self):
        """
        Map the prediction values to the original entries in the prediction dataset.
        """
        self.prediction_map = pd.DataFrame(self.prediction_location)
        self.prediction_map['label'] = self.prediction

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

        Raises:
            ValueError: If the model type is not supported.
            FileNotFoundError: If the filepath does not exist.
            TypeError: If the target_column is not a string, the features_list
                is not a list, or the features_list contains non-string elements
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
        Reconfigure the model setup with new parameters. Each model has its own set of parameters.
        The parameters that can be passed to the model are the same as the ones available in the
        scikit-learn models. The parameters can be found in the documentation of the scikit-learn.

        Input:
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
        Evaluate the performance of the trained model. It calls for the classification report
        function from scikit-learn and returns a dictionary containing the classification report.
        Printing the classification report is optional by setting the show parameter to True.
    
        Input:
            show (bool):  Default is False. If True, print the classification report.
    
        Return:
            report (dict): A dictionary containing the classification
        """
        if not isinstance(show, bool):
            raise TypeError('Plot must be a boolean.')
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.x_test)
        elif self.y_pred_test is np.ndarray:
            self.y_pred = self.y_pred
        if show is True:
            print(classification_report(self.y_test, self.y_pred, output_dict=False))
        self.report = classification_report(self.y_test, self.y_pred, output_dict=True)
        return self.report

    def predict(self, data):
        """
        This function makes predictions using the current trained model. It can take in a pandas
        DataFrame, an xarray Dataset, or a file path to a CSV or NetCDF file. The function will
        return the predicted values and store the input data for future reference.

        Input:
            x (array-like): The input features for making predictions.
            It can be a pandas DataFrame, an xarray Dataset, or a file path to a CSV or NetCDF file.

        Returns:
            array: The predicted values.

        Raises:
            FileNotFoundError: If the file path does not exist.
            ValueError: If the file format is not supported.
        """
        warnings.filterwarnings("ignore", message="numpy.ndarray size changed,"\
                                        " may indicate binary incompatibility")
        coordinate_columns = ['coord', 'coordinates', 'position', 'pos']
        type_of_data = type(data)
        # Read the data and extract the features for a pandas dataframe as input
        if type_of_data == pd.DataFrame:
            data_to_predict = data[self.features_list]
            self.prediction_dataset_size = data_to_predict.shape[0]
            location_columns = data[
                [col for col in data.columns
                 if any(word in col.lower() for word in coordinate_columns)]
                ]
            self.prediction_location = location_columns
        # Extract the features for an xarray dataset as input
        elif type_of_data == xr.Dataset:
            data_to_predict = data.to_dataframe()[self.features_list]
            self.prediction_dataset_size = data_to_predict.shape[0]
            location_columns = data[
                [col for col in data.columns
                 if any(word in col.lower() for word in coordinate_columns)]
                ]
            self.prediction_location = location_columns
        # Read the data from a file and extract the features
        elif type_of_data == str:
            if not os.path.isfile(data):
                raise FileNotFoundError(f"File '{data}' does not exist.")
            # Read the data from a CSV file
            elif data.endswith('.csv'):
                csv_df = pd.read_csv(data, header=0)
                data_to_predict = csv_df[self.features_list]
                self.prediction_dataset_size = csv_df.shape[0]
                location_columns = csv_df[
                    [col for col in csv_df.columns
                     if any(word in col.lower() for word in coordinate_columns)]
                    ]
                self.prediction_location = location_columns
            # Read the data from a NetCDF file
            elif data.endswith('.nc'):
                ds = xr.open_dataset(data)
                features_bytes = ds.features.values
                features_string = np.char.decode(features_bytes).tolist()
                features_list = features_string.split('0')
                features_list.insert(0, "longitude")
                features_list.insert(0, "latitude")
                values = ds.Result.values
                if values.shape[1] != len(features_list):
                    raise ValueError("The number of features does not match the " \
                                     "number of columns in the values array")
                df = pd.DataFrame(values, columns=features_list)
                self.prediction_dataset_size = df.shape[0]
                self.prediction_location = df[['latitude', 'longitude']]
                data_to_predict = df[self.features_list]
            else:
                raise ValueError("Invalid file format. Supported formats are CSV and NetCDF.")
        else:
            raise ValueError("Unsupported data type. Supported types are pandas DataFrame, " \
                             "xarray Dataset, and file path to CSV or NetCDF file.")
        self.prediction_object_type = type_of_data
        self.prediction_object = data
        self.prediction = self.model.predict(data_to_predict)
        self.__mapping()
        return self.prediction

    def save_model(self, filepath):
        """
        Save the trained model to a file.

        Input:
            filepath (str): The filepath to save the model to.

        Raises:
            ValueError: If the filepath is not a string.
        """
        if not isinstance(filepath, str):
            raise ValueError('Filepath must be a string.')
        joblib.dump(self, filepath)

    def generate_heatmap(self, filepath):
        """
        Save the heatmap of the dataset to a file.

        Input:
            filepath (str): The filepath to save the heatmap to.

        Raises:
            TypeError: If the filepath is not a string.
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
