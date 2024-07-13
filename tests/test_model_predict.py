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
        self.data_filepath_nc = "./testcase_data/prediction.nc"
        self.data_filepath_csv = "./testcase_data/training.csv"

    def test_predict_for_pandas_dataset(self):
        """
        Test the predict method for a random forest model.
        """
        dataset = pd.read_csv(self.data_filepath_csv, header=0)
        self.random_forest.predict(dataset)
        self.assertIsInstance(self.random_forest.last_prediction, np.ndarray)

    def test_predict_for_csv_file(self):
        """
        Test the predict method for a random forest model.
        """

        self.random_forest.predict(self.data_filepath_csv)
        self.assertIsInstance(self.random_forest.last_prediction, np.ndarray)

    def test_predict_for_nc_file(self):
        """
        Test the predict method for a random forest model.
        """
        self.random_forest.predict(self.data_filepath_nc)
        self.assertIsInstance(self.random_forest.last_prediction, np.ndarray)

if __name__ == '__main__':
    unittest.main(verbosity=2)
