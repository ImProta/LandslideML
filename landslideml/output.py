"""
This module provides functions for comparing different features and results from machine learning
models created with the LandslideML package.
"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from landslideml.model import MlModel

def __create_metrics_df(models: list[MlModel]) -> pd.DataFrame:
    """
    Create a DataFrame containing the metrics of the models for comparison.

    Input:
        models (list[MlModel]): A list of models to compare.

    Returns:
        pd.DataFrame: A DataFrame containing the metrics of the models for comparison.
    """
    metrics = ['precision', 'recall', 'f1-score', 'support']
    combined_data = []

    for model in models:
        model_name = model.type
        report = model.report
        for class_label, class_metrics in report.items():
            if class_label in ['accuracy']:
                combined_data.append({
                    'Model': model_name,
                    'Class': class_label,
                    'Metric': class_label,
                    'Value': class_metrics
                })
            else:
                for metric in metrics:
                    combined_data.append({
                        'Model': model_name,
                        'Class': class_label,
                        'Metric': metric,
                        'Value': class_metrics[metric]
                    })
    gathered_df = pd.DataFrame(combined_data)
    return gathered_df

def generate_heatmap(model: MlModel, filepath: str=None):
    """
    Generate the heatmap of a model for all the features present in the model dataset. The heatmap
    shows the correlation between all the features in the dataset. The heatmap can be saved to a 
    file if a filepath is provided. Otherwise, the heatmap is displayed.

    Input:
        model (MlModel): The model for which the heatmap is to be generated.
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
    keywords = ['coord', 'loc', 'location', 'coordinates', 'pos', 'position']
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

def compare_metrics(*models: MlModel, filepath: str = None, palette: str = 'dark:skyblue'):
    """
    Compare the metrics of two or more different models. Takes as input a variable number of models.
    The metrics in the classification report are compared for each model and displayed in a barplot.
    The barplot can be saved to a file if a filepath is provided. Otherwise, the barplot is 
    displayed.

    Input:
        models (MlModel): A variable number of models to compare.
        filepath (str): The filepath to save the comparison plot to. Default is None.
        palette (str): The color palette to use for the plot. Default is 'dark:skyblue'.

    Raises:
        warnings: If the models have different features.
    """
    if not isinstance(filepath, str) and filepath is not None:
        raise TypeError('Filepath must be a string.')
    if len(models) < 2:
        raise ValueError('At least two models are required for comparison.')

    reference_features = models[0].features_list
    for i, model in enumerate(models[1:], start=2):
        if model.features_list != reference_features:
            warnings.warn(f'Model {i} has different features from the first model.')

    metrics_df = __create_metrics_df(models)
    model_types = [model.type for model in models]
    model_labels = models[0].dataset[models[0].target_column].unique()
    labels_list = [str(item) for item in model_labels]
    labels_list.append('accuracy')
    classes = labels_list
    metric_order = ['precision', 'recall', 'f1-score', 'accuracy']

    filtered_df = metrics_df[(metrics_df['Model'].isin(model_types))
                             & (metrics_df['Class'].isin(classes))]
    pivot_df = filtered_df.pivot_table(index=['Model', 'Class', 'Metric'],
                                       values='Value').reset_index()

    fig, axes = plt.subplots(nrows=len(metric_order), ncols=1, figsize=(12, 12), sharey='row')

    for i, metric in enumerate(metric_order):
        if metric != 'accuracy':
            plot_data = pivot_df[pivot_df['Metric'] != 'accuracy']
        else:
            plot_data = pivot_df[pivot_df['Metric'] == 'accuracy']
        sns.barplot(x='Model',
                    y='Value',
                    hue='Class',
                    data=plot_data[plot_data['Metric'] == metric],
                    ax=axes[i],
                    palette=palette)
        axes[i].set_title(f'{metric.capitalize()} - Labels')
        axes[i].set_xlabel('')
        axes[i].set_ylabel(metric.capitalize())
        if metric != 'accuracy':
            handles, labels = axes[i].get_legend_handles_labels()
            axes[i].legend(handles=handles[:2], labels=labels[:2])
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
