"""
This file tests the workflow for generating a susceptibility map with the LandslideML package.
"""

import os
import unittest
from unittest.mock import patch
import landslideml as lsm


class TestPlotMap(unittest.TestCase):
    """
    Test the plot_map method in the LandslideML package for a GBM and Random Forest.
    Test cases:
    """

    def setUp(self):
        """
        Set up the test environment for the test cases.
        Generate three different models: Random Forest model with test size 0.25, another with
        test size 0.35, and a SVM model with test size 0.25.
        """

        self.filepath = "./testcase_data/training.csv"
        self.features = [
            "alti",
            "slope",
            "aspect",
            "bulk_density",
            "sand",
            "usda_classes",
            "silt",
            "clay",
            "coarse_fragments",
            "avail_water_capacity",
            "land_cover",
            "tree_cover_density",
            "obs_period_max",
            "saturated_water_content",
            "alpha_mrc",
            "n_mrc",
        ]
        self.type_gbm = "GBM"
        self.type_rf = "RandomForest"
        self.target = "label"
        self.test_size = 0.25
        self.filepath_prediction = "testcase_data/sample_prediction.nc"
        self.shapefile_path = "testcase_data/shapefile.shp"
        self.map_filepath = "testcase_data/map"

        self.gbm = lsm.generate_model(
            self.filepath, self.type_gbm, self.features, self.target, self.test_size
        )
        self.gbm.setup(n_estimators=100, max_depth=4, warm_start=True)
        self.gbm.evaluate_model()
        self.gbm.predict(self.filepath_prediction)

        self.rf = lsm.generate_model(
            self.filepath, self.type_rf, self.features, self.target, self.test_size
        )
        self.rf.setup(n_estimators=100, max_depth=10)
        self.rf.evaluate_model()
        self.rf.predict(self.filepath_prediction)

        self.rf_2 = lsm.generate_model(
            self.filepath,
            self.type_rf,
            self.features,
            self.target,
            self.test_size + 0.1,
        )
        self.rf_2.setup(n_estimators=100, max_depth=10)
        self.rf_2.evaluate_model()
        self.rf_2.predict(self.filepath_prediction)

    def tearDown(self):
        """
        Clean up generated model files after each test method.
        """

        file_dir = os.path.dirname(self.map_filepath)
        file_name = os.path.basename(self.map_filepath)
        similar_files = [f for f in os.listdir(file_dir) if f.startswith(file_name)]
        for file in similar_files:
            os.remove(os.path.join(file_dir, file))

    def test_plot_map_with_single_model(self):
        """
        Test the plot_map method for a GBM model.
        It verifies if the map file is generated.
        """

        lsm.plot_map(
            self.gbm, filepath=self.map_filepath, shp_filepath=self.shapefile_path
        )
        self.assertEqual(self.gbm.type, "GBM")
        self.assertTrue(os.path.basename(self.map_filepath))

    def test_plot_map_wrong_filepath(self):
        """
        Test the plot_map method with a wrong filepath.
        It verifies if the TypeError is raised.
        """

        with self.assertRaises(TypeError):
            lsm.plot_map(self.gbm, filepath=1, shp_filepath=self.shapefile_path)

    def test_plot_map_different_features(self):
        """
        Test the plot_map method with different features.
        It verifies if the warning is raised as expected.
        """

        rf_dif = lsm.generate_model(
            self.filepath, self.type_rf, self.features[:2], self.target, self.test_size
        )
        rf_dif.setup(n_estimators=100, max_depth=10)
        rf_dif.evaluate_model()
        rf_dif.predict(self.filepath_prediction)

        with self.assertWarns(UserWarning) as cm:
            lsm.plot_map(
                self.rf,
                rf_dif,
                filepath=self.map_filepath,
                shp_filepath=self.shapefile_path,
            )

        self.assertIn(
            f"Models with type '{self.rf.type}' and test size '{rf_dif.test_size}'"
            " have different features.",
            str(cm.warning),
        )

    def test_plot_map_unavailable_prediction_map(self):
        """
        Test the plot_map method with a model without prediction_map attribute.
        It verifies if the AttributeError is raised as expected.
        """

        rf_no_map = lsm.generate_model(
            self.filepath, self.type_rf, self.features, self.target, self.test_size
        )
        rf_no_map.setup(n_estimators=100, max_depth=10)
        rf_no_map.evaluate_model()
        with self.assertRaises(AttributeError):
            lsm.plot_map(
                self.gbm,
                rf_no_map,
                filepath=self.map_filepath,
                shp_filepath=self.shapefile_path,
            )

    def test_plot_map_different_prediction_map(self):
        """
        Test the plot_map method for models with different prediction maps.
        It verifies if the warning is raised as expected.
        """

        rf_dif_map = lsm.generate_model(
            self.filepath, self.type_rf, self.features, self.target, self.test_size
        )
        rf_dif_map.setup(n_estimators=100, max_depth=10)
        rf_dif_map.evaluate_model()
        rf_dif_map.predict(self.filepath_prediction)
        rf_dif_map.prediction_map = rf_dif_map.prediction_map[50:]
        with self.assertWarns(UserWarning) as cm:
            lsm.plot_map(
                self.rf,
                rf_dif_map,
                filepath=self.map_filepath,
                shp_filepath=self.shapefile_path,
            )

        self.assertIn(
            "The models have different prediction_map structures.",
            str(cm.warning),
        )

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_map_saving(self, mock_savefig, mock_show):
        """
        Test the plot_map method for a GBM and Random Forest model.
        It verifies if the map file is generated.
        """

        lsm.plot_map(
            self.gbm,
            self.rf,
            filepath=self.map_filepath,
            shp_filepath=self.shapefile_path,
        )
        self.assertEqual(self.gbm.type, "GBM")
        self.assertEqual(self.rf.type, "RandomForest")
        self.assertTrue(os.path.basename(self.map_filepath))
        mock_savefig.assert_called()
        mock_show.assert_not_called()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_plot_map_saving_three_models(self, mock_savefig, mock_show):
        """
        Test the plot_map method for a GBM and two Random Forest model.
        It verifies if the map file is generated.
        """

        lsm.plot_map(
            self.gbm,
            self.rf,
            self.rf_2,
            filepath=self.map_filepath,
            shp_filepath=self.shapefile_path,
        )
        self.assertEqual(self.gbm.type, "GBM")
        self.assertEqual(self.rf.type, "RandomForest")
        self.assertTrue(os.path.basename(self.map_filepath))
        mock_savefig.assert_called()
        mock_show.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
