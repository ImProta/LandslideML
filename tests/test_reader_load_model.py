"""
Tests for the saving_model method in the MlModel class.
"""

import unittest
import os
from landslideml.reader import generate_model, load_model

class TestLoadModel(unittest.TestCase):
    """
    Tests for the load_model method in the MlModel class.
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
        self.model_filepath = 'tests/TestModelRandomForest2024.pkl'
        self.random_forest.save_model(self.model_filepath)

    def tearDown(self):
        """
        Clean up after each test method.
        """
        if os.path.exists(self.model_filepath):
            os.remove(self.model_filepath)

    def test_load_model(self):
        """
        Test the load_model method for a random forest model.
        """
        random_forest_loaded = load_model(self.model_filepath)
        self.assertEqual(self.random_forest.kwargs, random_forest_loaded.kwargs)
        self.assertEqual(self.random_forest.type, random_forest_loaded.type)
        self.assertEqual(self.random_forest.target_column, random_forest_loaded.target_column)
        self.assertEqual(self.random_forest.features_list, random_forest_loaded.features_list)
        self.assertEqual(self.random_forest.test_size, random_forest_loaded.test_size)

    def test_load_model_wrong_input(self):
        """
        Test the load_model method for a random forest model with wrong input.
        """
        wrong_filepath = 123
        with self.assertRaises(ValueError):
            load_model(wrong_filepath)

    def test_load_model_file_not_found(self):
        """
        Test the load_model method for a random forest model with file not found.
        """
        non_existent_filepath = 'tests/filethatdoesnotexist.pkl'
        with self.assertRaises(FileNotFoundError):
            load_model(non_existent_filepath)


if __name__ == '__main__':
    unittest.main(verbosity=2)
