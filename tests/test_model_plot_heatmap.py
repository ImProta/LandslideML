"""
Tests for the saving_model method in the MlModel class.
"""

import os
import unittest
from landslideml.reader import generate_model

class TestPlotHeatmap(unittest.TestCase):
    """
    This class is used to test the method "plot_heatmap" inside the MlModel class.
    """

    def setUp(self):
        """
        Set up the test environment for the test cases. 
        Define the workflow for generating the random forest model.
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
        self.random_forest.evaluate_model()
        self.heatmap_filepath = 'tests/heatmap_dataset.png'

    def tearDown(self):
        """
        Clean up the test environment after the test cases are run.
        """
        if os.path.exists(self.heatmap_filepath):
            os.remove(self.heatmap_filepath)

    def test_plot_heatmap(self):
        """
        Test the plot_heatmap method for a random forest model.
        """
        self.random_forest.generate_heatmap(self.heatmap_filepath)
        self.assertTrue(os.path.exists(self.heatmap_filepath))

    def test_plot_heatmap_invalid_input(self):
        """
        Test the plot_heatmap method for a random forest model with invalid input.
        """
        with self.assertRaises(TypeError):
            self.random_forest.generate_heatmap(True)
        with self.assertRaises(TypeError):
            self.random_forest.generate_heatmap(1)

if __name__ == '__main__':
    unittest.main(verbosity=2)
