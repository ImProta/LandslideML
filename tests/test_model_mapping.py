"""
This file is used to test the method "evaluate_model" inside the MlModel class.
"""

import unittest
from landslideml.reader import generate_model


class TestEvaluateMappingMethod(unittest.TestCase):
    """
    A test case class for evaluating the __mapping method of a random forest model.
    Test cases: 
    - test_evaluate_model_no_plot: Testing the generated report of the evaluate_model method.
    - test_evaluate_model_wrong_parameters: Testing the method when inputting wrong parameters.
    - test_evaluate_model_warning_message: Testing the method not initializing a prediction data.
    """

    def setUp(self):
        """
        Set up the test environment for the test cases.
        Define the filepath, features, model type, and target for the test cases.
        Create a Random Forest model, set up the model parameters and makes a prediction
        with a nc file.
        """
        self.filepath = "./testcase_data/training.csv"
        self.prediction_data = "./testcase_data/prediction_cropped.nc"
        self.features = ["tree_cover_density", "alti", "slope", "clay"]
        self.model_type = "RandomForest"
        self.target = "label"
        self.test_size = 0.2
        self.random_forest = generate_model(
            self.filepath, self.model_type, self.features, self.target, self.test_size
        )
        self.random_forest.setup(n_estimators=100, max_depth=10)
        self.random_forest.predict(self.prediction_data)

    def test_mapping_generation(self):
        """
        Test the __mapping method for a random forest model.
        It verifies the shape of the prediction map and the labels in the prediction map.
        """
        shape_of_result = self.random_forest.prediction_map.shape
        labels = self.random_forest.prediction_map["label"]
        self.assertEqual(shape_of_result[0], self.random_forest.prediction_dataset_size)
        self.assertEqual(shape_of_result[1], 3)
        self.assertTrue(all(label in [0, 1] for label in labels))


if __name__ == "__main__":
    unittest.main(verbosity=3)
