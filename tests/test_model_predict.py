"""
Tests for the predict method in the MlModel class.
"""

import unittest
import numpy as np
import pandas as pd
from landslideml.reader import generate_model

class TestPredictMethod(unittest.TestCase):
    """
    Tests for the predict method in the MlModel class.
    """

    def setUp(self):
        self.filepath = "./testcase_data/training.csv"
        self.features = ['tree_cover_density','alti','slope', 'clay']
        self.model_type = 'RandomForest'
        self.target = 'label'
        self.test_size = 0.2
        self.random_forest = generate_model(self.filepath,
                                            self.model_type,
                                            self.features,
                                            self.target,
                                            self.test_size)
        self.random_forest.setup(n_estimators=100, max_depth=10)
        self.data_filepath_nc = "./testcase_data/prediction_cropped.nc"
        self.data_filepath_csv = "./testcase_data/training.csv"

    def test_predict_for_pandas_dataset(self):
        """
        Test the predict method for a random forest model with a pandas dataframe as input.
        """
        dataset = pd.read_csv(self.data_filepath_csv, header=0)
        self.random_forest.predict(dataset)
        self.assertEqual(self.random_forest.last_prediction_object_type, pd.DataFrame)

        self.assertIsInstance(self.random_forest.last_prediction, np.ndarray)
        self.assertEqual(self.random_forest.last_prediction.shape[0],
                         self.random_forest.last_prediction_dataset_size)

    def test_predict_for_csv_file(self):
        """
        Test the predict method for a random forest model with csv file as input.
        """

        self.random_forest.predict(self.data_filepath_csv)
        self.assertEqual(self.random_forest.last_prediction_object_type, str)
        self.assertEqual(self.random_forest.last_prediction.shape[0],
                         self.random_forest.last_prediction_dataset_size)

    def test_predict_for_nc_file(self):
        """
        Test the predict method for a random forest model with nc file as input.
        """
        self.random_forest.predict(self.data_filepath_nc)
        self.assertEqual(self.random_forest.last_prediction_object_type, str)
        self.assertIsInstance(self.random_forest.last_prediction, np.ndarray)
        self.assertEqual(self.random_forest.last_prediction.shape[0],
                         self.random_forest.last_prediction_dataset_size)

    def test_predict_for_wrong_input(self):
        """
        Test the predict method for a random forest model with a wrong input.
        """
        with self.assertRaises(ValueError):
            self.random_forest.predict(100)

    def test_predict_for_unsupported_file_format(self):
        """
        Test the predict method for a random forest model with an unsupported file format.
        """
        with self.assertRaises(ValueError):
            self.random_forest.predict("./testcase_data/prediction_cropped.txt")

    def test_predict_for_non_existent_file(self):
        """
        Test the predict method for a random forest model with a non-existent file.
        """
        with self.assertRaises(FileNotFoundError):
            self.random_forest.predict("./testcase_data/non_existent.csv")

if __name__ == '__main__':
    unittest.main(verbosity=2)
