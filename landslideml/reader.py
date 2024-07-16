"""
This module provides functions for reading and loading data in the LandslideML project.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from landslideml.model import MlModel
from .config import VALID_MODELS

def generate_model(filepath: str,
                   model_type: str,
                   features: list,
                   target: str,
                   test_size: float) -> MlModel:
    """
    Create a machine learning model for landslide prediction.

    Input:
        filepath (str): The file path of the dataset.
        model_type (str): The type of machine learning model to generate. 
                        Select from 'RandomForest', 'SVM', 'GBM'.
        features (list): The list of feature names.
        target (str): The target variable name.

    Returns:
        MlModel: An instance of the MlModel class representing the trained machine learning model.
        
    Raises:
        ValueError: If an invalid model type is selected.
    """
    # Generate a match case structure to return the corresponding MlModel
    match model_type:
        case "RandomForest":
            # Get the allowed input arguments for RandomForestClassifier
            rfc_args = RandomForestClassifier().get_params()
            return MlModel(filepath=filepath,
                           model_type=model_type,
                           features_list=features,
                           target_column=target,
                           test_size=test_size,
                           kwargs=rfc_args)
        case "SVM":
            svm_args = SVC().get_params()
            return MlModel(filepath=filepath,
                              model_type=model_type,
                              features_list=features,
                              target_column=target,
                              test_size=test_size,
                              kwargs=svm_args)
        case "GBM":
            bgm_args = GradientBoostingClassifier().get_params()
            return MlModel(filepath=filepath,
                              model_type=model_type,
                              features_list=features,
                              target_column=target,
                              test_size=test_size,
                              kwargs=bgm_args)
        case _:
            raise ValueError(f"Select a model from the list: {', '.join(VALID_MODELS)}.")

def load_model(filepath:str):
    """
    Load a saved model from a binary file with joblib.

    Input:
        filepath (str): The file path of the saved model.

    Returns:
        object: The loaded model object.

    Raises:
        ValueError: If the file path is not a string.
        FileNotFoundError: If the file path does not exist.
    """
    if not isinstance(filepath, str):
        raise ValueError("The file path must be a string.")
    try:
        model = joblib.load(filepath)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Model file not found at {filepath}.") from exc
    return model
