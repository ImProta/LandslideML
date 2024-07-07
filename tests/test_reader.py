"""
This module is used to test the reader module of the LandslideML package.
"""

import unittest
from landslideml.reader import generate_model
from landslideml.model import MlModel
from landslideml import VALID_MODELS


class TestReaderMethod(unittest.TestCase):
    """
    Test the reader module for different input parameters for the Random Forest ML model. 
    """

    def setUp(self):
        """
        Set up the test environment for the test cases. 
        Define the filepath and features for the test cases.
        """
        self.valid_models = VALID_MODELS
        self.filepath = "./testcase_data/training.csv"
        self.features = ['slope', 'clay']
        self.model_type = 'RandomForest'
        self.target = 'label'

    def test_generate_model_random_forest(self):
        """
        Test the generate_model function for the random forest.
        """

        random_forest = generate_model(self.filepath, self.model_type, self.features, self.target)
        self.assertIsInstance(random_forest, MlModel)
        self.assertEqual(random_forest.filepath, self.filepath)
        self.assertEqual(random_forest.type, self.model_type)
        self.assertEqual(random_forest.features_list, self.features)
        self.assertEqual(random_forest.target_column, self.target)

    def test_valid_model_types_(self):
        """
        Test the generate_model function for valid model types.
        """

        for model_type in self.valid_models:
            random_forest = generate_model(self.filepath, model_type, self.features, self.target)
            self.assertIsInstance(random_forest, MlModel)
            self.assertEqual(random_forest.type, model_type)


    def test_invalid_model_type(self):
        """
        Test the generate_model function for an invalid model type.
        """

        invalid_model_type = 'InvalidModelType'
        with self.assertRaises(ValueError):
            generate_model(self.filepath, invalid_model_type, self.features, self.target)

if __name__ == '__main__':
    unittest.main(verbosity=2)
