"""
This file tests the workflow for generating a heatmap with the LandslideML package.
"""

import os
import unittest
from unittest.mock import patch
import landslideml as lsm


class TestGenerateHeatmap(unittest.TestCase):
    """
    Test the plot_heatmap method in the LandslideML package for a Random Forest model.
    Test cases:
    - test_plot_heatmap_with_filepath: Test the plot_heatmap method with a filepath.
    - test_plot_heatmap_without_filepath: Test the plot_heatmap method without a filepath.
    - test_plot_heatmap_with_invalid_filepath: Test the plot_heatmap method with
        an invalid filepath.
    - test_plot_heatmap_with_invalid_model_input: Test the plot_heatmap method with
        an invalid model input.
    """

    def setUp(self):
        """
        Set up the test environment for the test cases.
        Define the filepath, features, model type, target, and data split for the test cases.
        Also, generate a Random Forest model and set up the model parameters.
        """
        self.filepath = "./testcase_data/training.csv"
        self.features = ["tree_cover_density", "alti", "slope", "clay"]
        self.model_type = "RandomForest"
        self.target = "label"
        self.test_size = 0.2
        self.random_forest = lsm.generate_model(
            self.filepath, self.model_type, self.features, self.target, self.test_size
        )
        self.random_forest.setup(n_estimators=100, max_depth=10)
        self.random_forest.evaluate_model()
        self.heatmap_filepath = "tests/heatmap_dataset.png"

    def tearDown(self):
        """
        Clean up generated model file after each test method.
        """
        if os.path.exists(self.heatmap_filepath):
            os.remove(self.heatmap_filepath)

    def test_plot_heatmap_with_filepath(self):
        """
        Test the plot_heatmap method for a random forest model when given a filepath.
        It verifies if the heatmap file is generated.
        """
        lsm.plot_heatmap(self.random_forest, self.heatmap_filepath)
        self.assertTrue(os.path.exists(self.heatmap_filepath))

    @patch("matplotlib.pyplot.savefig")
    @patch("seaborn.heatmap")
    @patch("matplotlib.pyplot.show")
    def test_plot_heatmap_without_filepath(
        self, mock_show, mock_heatmap, mock_savefig
    ):
        """
        Test the plot_heatmap method for a random forest model without giving a filepath.
        It verifies if the heatmap is shown but not saved.
        """
        lsm.plot_heatmap(self.random_forest)
        mock_heatmap.assert_called_once()
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()

    def test_plot_heatmap_with_invalid_filepath(self):
        """
        Test the plot_heatmap method with an invalid filepath input.
        It verifies that the method raises a TypeError when the input is not a string.
        """
        with self.assertRaises(TypeError):
            lsm.plot_heatmap(self.random_forest, 123)

    def test_plot_heatmap_with_invalid_model_input(self):
        """
        Test the plot_heatmap method with an invalid model input.
        It verifies that the method raises a TypeError when the input is not a MlModel object.
        """
        with self.assertRaises(TypeError):
            lsm.plot_heatmap(123)


if __name__ == "__main__":
    unittest.main(verbosity=2)
