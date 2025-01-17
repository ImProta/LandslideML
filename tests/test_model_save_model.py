"""
this file tests the save_model method in the MlModel class.
"""

import unittest
import os
from landslideml.reader import generate_model


class TestSaveModelMethod(unittest.TestCase):
    """
    Test the save_model method in the MlModel class for a Random Forest model.
    Test cases:
    - test_evaluate_model_save_model: Test the save_model method for a random forest model.
    - test_evaluate_wrong_input: Test the save_model method for a random forest model with wrong input.
    """

    def setUp(self):
        """
        Set up the test environment for the test cases.
        Define the filepath, features, model type, target, and data split for the test cases.
        Generates and sets up a Random Forest model. 
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
        self.random_forest.evaluate_model()

    def test_evaluate_model_save_model(self):
        """
        Test the save_model method for a random forest model.
        It verifies if the model file is saved.
        """
        model_filepath = "tests/TestModelRandomForest.pkl"
        self.random_forest.save_model(model_filepath)
        self.assertTrue(os.path.isfile(model_filepath))
        os.remove(model_filepath)

    def test_evaluate_wrong_input(self):
        """
        Test the save_model method for a random forest model with wrong input.
        It verifies if the method raises a ValueError.
        """
        filepath = 123
        with self.assertRaises(ValueError):
            self.random_forest.save_model(filepath)


if __name__ == "__main__":
    unittest.main(verbosity=2)
