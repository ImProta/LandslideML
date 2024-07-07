"""
Tests for the saving_model method in the MlModel class.
"""

import unittest
import os
from landslideml.reader import generate_model

class TestSaveModelMethod(unittest.TestCase):
    """
    Tests for the save_model method in the MlModel class.
    """

    def setUp(self):
        self.filepath = "./testcase_data/training.csv"
        self.features = ['tree_cover_density','alti','slope', 'clay']
        self.model_type = 'RandomForest'
        self.target = 'label'
        self.random_forest = generate_model(self.filepath,
                                            self.model_type,
                                            self.features,
                                            self.target)
        self.random_forest.setup(n_estimators=100, max_depth=10)
        self.random_forest.evaluate_model()

    def test_evaluate_model_save_model(self):
        """
        Test the save_model method for a random forest model.
        """
        model_filepath = 'tests/TestModelRandomForest2024.pkl'
        self.random_forest.save_model(model_filepath)
        self.assertTrue(os.path.isfile(model_filepath))
        os.remove(model_filepath)

    def test_evaluate_wrong_input(self):
        """
        Test the save_model method for a random forest model with wrong input.
        """
        filepath = 123
        with self.assertRaises(ValueError):
            self.random_forest.save_model(filepath)

if __name__ == '__main__':
    unittest.main(verbosity=2)
