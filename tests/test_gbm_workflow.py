"""
This file is used to evaluate the workflow for generating a Supporting Vector Machine (gbm).
Test cases: 

"""
import os
import unittest

from sklearn.ensemble import GradientBoostingClassifier as GBM
import landslideml as lsm

class TestGBMModel(unittest.TestCase):
    """
    A test case class for evaluating the gbm model workflow.
    """

    def setUp(self):
        """
        Set up the test environment for the test cases.
        Define the filepath, features, model type, and target for the test cases.
        """
        self.filepath = "./testcase_data/training.csv"
        self.features = ['tree_cover_density','alti','bulk_density', 'slope', 'clay', 'sand']
        self.model_type = 'GBM'
        self.target = 'label'
        self.test_size = 0.25
        self.gbm = lsm.generate_model(self.filepath,
                                     self.model_type,
                                     self.features,
                                     self.target,
                                     self.test_size)
        self.gbm.setup(n_estimators=100, learning_rate=0.2, min_samples_leaf=5, max_depth = 4)
        self.save_model_path = "testcase_data/gbm_model.pkl"
        self.gbm.predict("testcase_data/prediction_cropped.nc")
        self.gbm.save_model(self.save_model_path)

    def tearDown(self):
        """
        Clean up after each test method.
        """
        if os.path.exists(self.save_model_path):
            os.remove(self.save_model_path)

    def test_gbm_model_attributes(self):
        """
        Test the attributes of the gbm model.
        """
        self.assertEqual(self.gbm.model_type, 'GBM')
        self.assertEqual(self.gbm.features_list, self.features)
        self.assertEqual(self.gbm.target_column, self.target)
        self.assertEqual(self.gbm.test_size, self.test_size)
        self.assertIsInstance(self.gbm.model, GBM)
        self.assertEqual(self.gbm.kwargs['n_estimators'], 100)
        self.assertEqual(self.gbm.kwargs['learning_rate'], 0.2)
        self.assertEqual(self.gbm.kwargs['min_samples_leaf'], 5)
        self.assertEqual(self.gbm.kwargs['max_depth'], 4)
        self.assertEqual(self.gbm.prediction_object, "testcase_data/prediction_cropped.nc")
        self.assertTrue(os.path.exists(self.save_model_path))

if __name__ == '__main__':
    unittest.main(verbosity=3)
