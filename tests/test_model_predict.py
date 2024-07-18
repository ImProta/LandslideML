"""
This file tests the predict method in the LandslideML package.
"""

import unittest
import numpy as np
import pandas as pd
from landslideml.reader import generate_model


class TestPredictMethod(unittest.TestCase):
    """
    Test the predict method in the LandslideML package for a Random Forest model.
    Test cases:
    - test_predict_for_pandas_dataset: Test the predict method with a pandas dataframe as input.
    - test_predict_for_csv_file: Test the predict method with a csv file as input.
    - test_predict_for_nc_file: Test the predict method with a nc file as input.
    - test_predict_for_wrong_input: Test the predict method with a wrong input.
    - test_predict_for_unsupported_file_format: Test the predict method with an unsupported file format.
    - test_predict_for_non_existent_file: Test the predict method with a non-existent file.
    
    """

    def setUp(self):
        """
        Set up the test environment for the test cases.
        Defines the filepath, features, model type, target, and data split for the test cases.
        Creates a Random Forest model and sets up the model parameters.
        """

        self.filepath = "./testcase_data/training.csv"
        self.features = ["tree_cover_density", "alti", "slope", "clay"]
        self.model_type = "RandomForest"
        self.target = "label"
        self.test_size = 0.2
        self.random_forest = generate_model(
            self.filepath, self.model_type, self.features, self.target, self.test_size
        )
        self.random_forest.setup(n_estimators=100, max_depth=10)
        self.data_filepath_nc = "./testcase_data/prediction_cropped.nc"
        self.data_filepath_csv = "./testcase_data/training.csv"

    def test_predict_for_pandas_dataset(self):
        """
        Test the predict method for a random forest model with a pandas dataframe as input.
        It verifies the prediction object type and the shape of the prediction. It should be a 
        numpy array with the same number of rows as the input dataset.
        """
        dataset = pd.read_csv(self.data_filepath_csv, header=0)
        self.random_forest.predict(dataset)
        self.assertEqual(self.random_forest.prediction_object_type, pd.DataFrame)

        self.assertIsInstance(self.random_forest.prediction, np.ndarray)
        self.assertEqual(
            self.random_forest.prediction.shape[0],
            self.random_forest.prediction_dataset_size,
        )

    def test_predict_for_csv_file(self):
        """
        Test the predict method for a random forest model with csv file as input.
        It verifies the prediction object type and the shape of the prediction. It should be a
        numpy array with the same number of rows as the input dataset.
        """

        self.random_forest.predict(self.data_filepath_csv)
        self.assertEqual(self.random_forest.prediction_object_type, str)
        self.assertEqual(
            self.random_forest.prediction.shape[0],
            self.random_forest.prediction_dataset_size,
        )

    def test_predict_for_nc_file(self):
        """
        Test the predict method for a random forest model with nc file as input.
        It verifies the prediction object type and the shape of the prediction. It should be a
        numpy array with the same number of rows as the input dataset.
        """
        self.random_forest.predict(self.data_filepath_nc)
        self.assertEqual(self.random_forest.prediction_object_type, str)
        self.assertIsInstance(self.random_forest.prediction, np.ndarray)
        self.assertEqual(
            self.random_forest.prediction.shape[0],
            self.random_forest.prediction_dataset_size,
        )

    def test_predict_for_wrong_input(self):
        """
        Test the predict method for a random forest model with a wrong input.
        It verifies that the method raises a ValueError when the input is not a pandas dataframe,
        csv file, or nc file.
        """
        with self.assertRaises(ValueError):
            self.random_forest.predict(100)

    def test_predict_for_unsupported_file_format(self):
        """
        Test the predict method for a random forest model with an unsupported file format.
        It verifies that the method raises a ValueError when the input is a file with an unsupported
        format.
        """
        with self.assertRaises(ValueError):
            self.random_forest.predict("./testcase_data/prediction_cropped.txt")

    def test_predict_for_non_existent_file(self):
        """
        Test the predict method for a random forest model with a non-existent file.
        It verifies that the method raises a FileNotFoundError when the input is a non-existent file.
        """
        with self.assertRaises(FileNotFoundError):
            self.random_forest.predict("./testcase_data/non_existent.csv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
