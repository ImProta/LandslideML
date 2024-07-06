"""
Reader module for reading in data.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from landslideml.model import MlModel
from landslideml import VALID_MODELS

def generate_model(filepath: str, model_type: str, features: list, target: str) -> MlModel:
    """
    Create a machine learning model for landslide prediction.

    Args:
        filepath (str): The file path of the dataset.
        model_type (str): The type of machine learning model to generate. 
                        Select from 'RandomForest', 'SVM', 'BGM'.
        features (list): The list of feature names.
        target (str): The target variable name.

    Returns:
        MlModel: An instance of the MlModel class representing the trained machine learning model.
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
                           kwargs=rfc_args)
        case "SVM":
            svm_args = SVC().get_params()
            return MlModel(filepath=filepath,
                              model_type=model_type,
                              features_list=features,
                              target_column=target,
                              kwargs=svm_args)
        case "GBM":
            bgm_args = GradientBoostingClassifier().get_params()
            return MlModel(filepath=filepath,
                              model_type=model_type,
                              features_list=features,
                              target_column=target,
                              kwargs=bgm_args)
        case _:
            raise ValueError(f"Select a model from the list: {', '.join(VALID_MODELS)}.")
