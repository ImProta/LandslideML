"""
Reader module for reading in data.
"""

from landslideml.model import MlModel

def generate_svm(filepath: str) -> MlModel:
    """
    Create a machine learning model for landslide prediction with SVM.
    """
    return MlModel(filepath=filepath, model_type='SVM')

def generate_random_forest(filepath: str, model_type:str, ) -> MlModel:
    """
    Create a machine learning model for landslide prediction with Random Forest.
    """
    return MlModel(filepath=filepath, model_type=model_type)

def generate_gbm(filepath: str) -> MlModel:
    """
    Create a machine learning model for landslide prediction with Gradient Boosting Machine.
    """
    return MlModel(filepath=filepath, model_type='GBM')
