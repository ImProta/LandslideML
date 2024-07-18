"""
This file tests the generate_model method in reader.py of the LandslideML package.
"""

import unittest
from landslideml.reader import generate_model
from landslideml.model import MlModel
from landslideml.config import VALID_MODELS


class TestReaderGenerateModel(unittest.TestCase):
    """
    Test the generate_model method with different inputs for a Random Forest Machine Learning model.
    Test cases:
    - test_generate_model_random_forest: Test the generate_model method to create a
        Random Forest model.
    - test_valid_model_types_: Test the generate_model method for valid model types in the
        VALID_MODELS list.
    - test_invalid_model_type: Test the generate_model method for an invalid model type.
    """

    def setUp(self):
        """
        Set up the test environment for the test cases.
        Define the filepath, features, model type, target, and data split for the test cases.
        """
        self.valid_models = VALID_MODELS
        self.filepath = "./testcase_data/training.csv"
        self.features = ["slope", "clay"]
        self.model_type = "RandomForest"
        self.target = "label"
        self.test_size = 0.2

    def test_generate_model_random_forest(self):
        """
        Test the generate_model method to create a Random Forest model.
        It verifies the model type, features, target, and test size of the generated model.
        """

        random_forest = generate_model(
            self.filepath, self.model_type, self.features, self.target, self.test_size
        )
        self.assertIsInstance(random_forest, MlModel)
        self.assertEqual(random_forest.filepath, self.filepath)
        self.assertEqual(random_forest.type, self.model_type)
        self.assertEqual(random_forest.features_list, self.features)
        self.assertEqual(random_forest.target_column, self.target)

    def test_valid_model_types_(self):
        """
        Test the generate_model method for valid model types in the VALID_MODELS list.
        It verifies the model type of the generated model.
        """

        for model_type in self.valid_models:
            random_forest = generate_model(
                self.filepath, model_type, self.features, self.target, self.test_size
            )
            self.assertIsInstance(random_forest, MlModel)
            self.assertEqual(random_forest.type, model_type)

    def test_invalid_model_type(self):
        """
        Test the generate_model method for an invalid model type (not in the VALID_MODELS list).
        It verifies the ValueError raised when an invalid model type is selected.
        """

        invalid_model_type = "InvalidModelType"
        with self.assertRaises(ValueError):
            generate_model(
                self.filepath,
                invalid_model_type,
                self.features,
                self.target,
                self.test_size,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
