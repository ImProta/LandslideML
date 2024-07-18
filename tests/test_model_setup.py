"""
This file tests the model setup method in the LandslideML package.
"""

import unittest
from landslideml.reader import generate_model


class TestModelSetup(unittest.TestCase):
    """
    Test the setup method in the MlModel class for a Random Forest model.
    Test cases:
    - test_model_setup_wrong_parameters: Test the setup method with wrong parameters.
    - test_model_setup: Test the setup method with correct parameters.
    """

    def setUp(self):
        """
        Set up the test environment for the test cases.
        Define the filepath, features, model type, target, and data split for the test cases.
        """
        self.filepath = "./testcase_data/training.csv"
        self.features = ["tree_cover_density", "alti", "slope", "clay"]
        self.model_type = "RandomForest"
        self.target = "label"
        self.test_size = 0.2

    def test_model_setup_wrong_parameters(self):
        """
        Test the setup method in the MlModel class for a Random Forest model when inputting
        """
        random_forest = generate_model(
            self.filepath, self.model_type, self.features, self.target, self.test_size
        )
        with self.assertRaises(ValueError):
            random_forest.setup(wrong_parameter="100", max_depth="10")

    def test_model_setup(self):
        """
        Test the setup method in the MlModel class for a Random Forest model when inputting
        the allowed parameters for the chosen model.
        """
        random_forest = generate_model(
            self.filepath, self.model_type, self.features, self.target, self.test_size
        )
        random_forest.setup(n_estimators=100, max_depth=10)
        self.assertEqual(random_forest.kwargs["n_estimators"], 100)
        self.assertEqual(random_forest.kwargs["max_depth"], 10)


if __name__ == "__main__":
    unittest.main(verbosity=3)
