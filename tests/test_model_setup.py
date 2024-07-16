"""
This module is used to test the model module of the LandslideML package.
"""

import unittest
from landslideml.reader import generate_model


class TestModelSetupMethod(unittest.TestCase):
    """
    Test the models module functionalities for the random forest model.
    """

    def setUp(self):
        """
        Set up the test environment for the test cases. 
        Define the filepath and features for the test cases.
        """
        self.filepath = "./testcase_data/training.csv"
        self.features = ['tree_cover_density','alti','slope', 'clay']
        self.model_type = 'RandomForest'
        self.target = 'label'
        self.test_size = 0.2

    def test_model_setup_wrong_parameters(self):
        """
        Test the generate_model function for the random forest when inputting wrong
        parameters in the setup function.
        """
        random_forest = generate_model(self.filepath,
                                       self.model_type,
                                       self.features,
                                       self.target,
                                       self.test_size)
        with self.assertRaises(ValueError):
            random_forest.setup(wrong_parameter='100', max_depth='10')

    def test_model_setup_correct_parameters(self):
        """
        Test the generate_model function for the random forest when inputting the correct 
        parameters in the setup function.
        """
        random_forest = generate_model(self.filepath,
                                       self.model_type,
                                       self.features,
                                       self.target,
                                       self.test_size)
        random_forest.setup(n_estimators=100, max_depth=10)
        self.assertEqual(random_forest.kwargs['n_estimators'], 100)
        self.assertEqual(random_forest.kwargs['max_depth'], 10)

if __name__ == '__main__':
    unittest.main(verbosity=3)
