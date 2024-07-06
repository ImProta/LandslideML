"""
This file is used to test the method "evaluate_model" inside the MlModel class.
Test cases: 
    - test_evaluate_model_no_plot: Testing the generated report of the evaluate_model method.
    - test_evaluate_model_wrong_parameters: Testing the method when inputting wrong parameters.
    - test_evaluate_model_warning_message: Testing the method not initializing a prediction data.
"""

import unittest
from landslideml.reader import generate_model


class TestEvaluateModelMethod(unittest.TestCase):
    """
    A test case class for evaluating the evaluate_model method of a random forest model.
    """

    def setUp(self):
        """
        Set up the test environment for the test cases. 
        Define the filepath, features, model type, and target for the test cases.
        """
        self.filepath = "./testcase_data/training.csv"
        self.features = ['tree_cover_density','alti','slope', 'clay']
        self.model_type = 'RandomForest'
        self.target = 'label'
        self.random_forest = generate_model(self.filepath,
                                            self.model_type,
                                            self.features,
                                            self.target)
        self.random_forest.setup(n_estimators=100, max_depth=10)

    def test_evaluate_model_no_plot(self):
        """
        Test the evaluate_model method for a random forest model.
        """
        self.random_forest.evaluate_model()
        self.assertIsNotNone(self.random_forest.report)
        self.assertIsInstance(self.random_forest.report, dict)


    def test_evaluate_model_wrong_parameters(self):
        """
        Test the evaluate_model method for a random forest model when inputting wrong parameters.
        """
        with self.assertRaises(TypeError):
            self.random_forest.evaluate_model(show=2)

    def test_evaluate_model_warning_message(self):
        """
        Test the evaluate_model method warning when a prediction dataset is not loaded.
        """
        with self.assertWarns(Warning):
            self.random_forest.evaluate_model()


if __name__ == '__main__':
    unittest.main(verbosity=3)
