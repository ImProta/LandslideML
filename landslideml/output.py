"""
This module provides functions for comparing different features and results from machine learning
models created with the LandslideML package.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from landslideml.model import MlModel

def generate_heatmap(model: MlModel, filepath: str=None):
    """
    Save the heatmap of the dataset to a file.

    Input:
        filepath (str): The filepath to save the heatmap to.

    Raises:
        TypeError: If the filepath is not a string.
    """
    if not isinstance(filepath, str) and filepath is not None:
        raise TypeError('Filepath must be a string.')
    plt.figure(figsize=(10, 8))
    numeric_data = model.dataset.select_dtypes(include=[float, int])
    if model.target_column in numeric_data.columns:
        numeric_data = numeric_data.drop(columns=[model.target_column])
    keywords = ['coord', 'loc', 'location', 'coordinates']
    columns_to_exclude = [col for col in numeric_data.columns
                            if any(keyword in col.lower() for keyword in keywords)]
    numeric_data = numeric_data.drop(columns=columns_to_exclude)
    sns.heatmap(numeric_data.corr(),
                xticklabels=numeric_data.columns,
                yticklabels=numeric_data.columns,
                annot=True,
                fmt='.3f',
                cmap='coolwarm')
    plt.title('Heatmap of Dataset Features')
    if filepath is not None:
        plt.savefig(filepath)
    elif filepath is None:
        plt.show()
