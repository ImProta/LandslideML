"""
Tests for the generate_heatmap method in the output module.
"""

import os
import unittest
from unittest.mock import patch
import landslideml as lsm

class TestGenerateHeatmapMethod(unittest.TestCase):
    """
    This class is used to test the method "generate_heatmap" in the output module.
    """

    def setUp(self):
        """
        Set up the test environment for the test cases.
        """
        self.filepath = "./testcase_data/training.csv"
        self.features = ['tree_cover_density','alti','slope', 'clay']
        self.model_type = 'RandomForest'
        self.target = 'label'
        self.test_size = 0.2
        self.random_forest = lsm.generate_model(self.filepath,
                                                self.model_type,
                                                self.features,
                                                self.target,
                                                self.test_size)
        self.random_forest.setup(n_estimators=100, max_depth=10)
        self.random_forest.evaluate_model()
        self.heatmap_filepath = 'tests/heatmap_dataset.png'

    def tearDown(self):
        """
        Clean up the test environment after the test cases are run.
        """
        if os.path.exists(self.heatmap_filepath):
            os.remove(self.heatmap_filepath)

    def test_generate_heatmap_with_filepath(self):
        """
        Test the generate_heatmap method for a random forest model with a filepath.
        """
        lsm.generate_heatmap(self.random_forest, self.heatmap_filepath)
        self.assertTrue(os.path.exists(self.heatmap_filepath))

    @patch('matplotlib.pyplot.savefig')
    @patch('seaborn.heatmap')
    @patch('matplotlib.pyplot.show')
    def test_generate_heatmap_without_filepath(self, mock_show, mock_heatmap, mock_savefig):
        """
        Test the generate_heatmap method for a random forest model without a filepath.
        """
        lsm.generate_heatmap(self.random_forest)
        mock_heatmap.assert_called_once()
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()

    def test_generate_heatmap_with_invalid_filepath(self):
        """
        Test the generate_heatmap method with an invalid filepath.
        """
        with self.assertRaises(TypeError):
            lsm.generate_heatmap(self.random_forest, 123)

    def test_generate_heatmap_with_invalid_model(self):
        """
        Test the generate_heatmap method with an invalid model.
        """
        with self.assertRaises(TypeError):
            lsm.generate_heatmap(123)

if __name__ == '__main__':
    unittest.main(verbosity=2)
