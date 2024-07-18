"""
This file tests the workflow for generating a Supporting Vector Machine (SVM) with 
the LandslideML package.
"""

import os
import unittest

from sklearn.svm import SVC
import landslideml as lsm


class TestSVMModel(unittest.TestCase):
    """
    Test the workflow for generating a Supporting Vector Machine (SVM) model with
        the LandslideML package.
    Test case:
    - test_svm_model_attributes: Test the attributes of the SVM model.
    """

    def setUp(self):
        """
        Set up the test environment for the test cases.
        Define the filepath, features, model type, target, and data split for the test cases.
        """
        self.filepath = "./testcase_data/training.csv"
        self.features = ["tree_cover_density", "alti", "slope", "clay", "sand"]
        self.model_type = "SVM"
        self.target = "label"
        self.test_size = 0.25
        self.svm = lsm.generate_model(
            self.filepath, self.model_type, self.features, self.target, self.test_size
        )
        self.svm.setup(degree=3, kernel="rbf", C=1.0, gamma="scale")
        self.save_model_path = "testcase_data/svm_model.pkl"
        self.svm.predict("testcase_data/prediction_cropped.nc")
        self.svm.save_model(self.save_model_path)

    def tearDown(self):
        """
        Clean up generated file after each test method.
        """
        if os.path.exists(self.save_model_path):
            os.remove(self.save_model_path)

    def test_svm_model_attributes(self):
        """
        Test the attributes of the object MlModel for the SVM model.
        It verifies the model type, features, target, test size, model, and prediction object
        of the generated model.
        """
        self.assertEqual(self.svm.type, "SVM")
        self.assertEqual(self.svm.features_list, self.features)
        self.assertEqual(self.svm.target_column, self.target)
        self.assertEqual(self.svm.test_size, self.test_size)
        self.assertIsInstance(self.svm.model, SVC)
        self.assertEqual(self.svm.kwargs["degree"], 3)
        self.assertEqual(self.svm.kwargs["kernel"], "rbf")
        self.assertEqual(self.svm.kwargs["C"], 1.0)
        self.assertEqual(self.svm.kwargs["gamma"], "scale")
        self.assertEqual(
            self.svm.prediction_object, "testcase_data/prediction_cropped.nc"
        )
        self.assertTrue(os.path.exists(self.save_model_path))


if __name__ == "__main__":
    unittest.main(verbosity=3)
