"""
Tests for the generate_heatmap method in the output module.
"""
import os
import unittest
from unittest.mock import patch
import landslideml as lsm

class TestCompareMetricsMethod(unittest.TestCase):
    """
    This class is used to test the method "generate_heatmap" in the output module.
    """

    def setUp(self):
        """
        Set up a Random Forest and a SVM models for comparison.
        Defines input parameters for the model (filepath, features, model_type, target, test_size)
        and generates a Random Forest and SVM with same parameters.
        """

        self.filepath = "./testcase_data/training.csv"
        self.features = ['tree_cover_density','alti','slope', 'clay']
        self.target = 'label'
        self.test_size = 0.2
        self.image_path = 'compare_metrics.png'

        self.random_forest = lsm.generate_model(self.filepath,
                                                'RandomForest',
                                                self.features,
                                                self.target,
                                                self.test_size)
        self.random_forest.setup(n_estimators=100, max_depth=10)
        self.random_forest.evaluate_model()

        self.svm = lsm.generate_model(self.filepath,
                                      'SVM',
                                      self.features,
                                      self.target,
                                      self.test_size)
        self.svm.setup(kernel='rbf', C=1)
        self.svm.evaluate_model()

        self.svm_2 = lsm.generate_model(self.filepath,
                                        'SVM',
                                        self.features + ['bulk_density'],
                                        self.target,
                                        self.test_size + 0.1)
        self.svm_2.setup(kernel='rbf', C=1)
        self.svm_2.evaluate_model()

    def tearDown(self):
        """
        Remove the image file if it exists.
        """
        if os.path.exists(self.image_path):
            os.remove(self.image_path)

    def test_compare_metrics_insufficient_models(self):
        """
        Test the compare_metrics method with only one model.
        """
        with self.assertRaises(ValueError):
            lsm.compare_metrics(self.random_forest)

    @patch('matplotlib.pyplot.savefig')
    @patch('seaborn.barplot')
    @patch('matplotlib.pyplot.show')
    def test_compare_metrics_different_attributes(self, mock_show, mock_barplot, mock_savefig):
        """
        Test the compare_metrics method with two models with different features.
        """
        with self.assertWarns(Warning):
            lsm.compare_metrics(self.random_forest, self.svm_2)
        mock_show.assert_called_once()
        mock_barplot.assert_called()
        mock_savefig.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_compare_metrics_same_attributes(self, mock_show):
        """
        Test the compare_metrics method with two models with the same features.
        """
        lsm.compare_metrics(self.random_forest, self.svm)
        mock_show.assert_called_once()

    def test_compare_metrics_filepath(self):
        """
        Test the compare_metrics method with a filepath argument.
        """
        lsm.compare_metrics(self.random_forest, self.svm, filepath=self.image_path)
        self.assertTrue(os.path.exists(self.image_path))

if __name__ == '__main__':
    unittest.main(verbosity=2)
