"""
This module provides functions for comparing different features and results from machine learning
models created with the LandslideML package.
"""

import warnings
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap
from landslideml.model import MlModel


def __create_metrics_df(models: list[MlModel]) -> pd.DataFrame:
    """
    Create a DataFrame containing the metrics of the models for comparison.

    Input:
        models (list[MlModel]): A list of models to compare.

    Returns:
        pd.DataFrame: A DataFrame containing the metrics of the models for comparison.
    """
    metrics = ["precision", "recall", "f1-score", "support"]
    combined_data = []

    for model in models:
        model_name = model.type
        report = model.report
        for class_label, class_metrics in report.items():
            if class_label in ["accuracy"]:
                combined_data.append(
                    {
                        "Model": model_name,
                        "Class": class_label,
                        "Metric": class_label,
                        "Value": class_metrics,
                    }
                )
            else:
                for metric in metrics:
                    combined_data.append(
                        {
                            "Model": model_name,
                            "Class": class_label,
                            "Metric": metric,
                            "Value": class_metrics[metric],
                        }
                    )
    gathered_df = pd.DataFrame(combined_data)
    return gathered_df


def __get_column_indices(columns, possible_columns):
    """
    Get the index of the first column in a list of columns that matches one of the possible columns.
    If no column matches, return None.

    Input:
        columns (list): A list of column names.
        possible_columns (list): A list of possible column names to match.

    Returns:
        int: The index of the first column that matches one of the possible columns.
        None: If no column matches any of the possible columns.
    """
    lower_columns = [col.lower() for col in columns]
    for possible_column in possible_columns:
        if possible_column in lower_columns:
            return columns[lower_columns.index(possible_column)]
    return None


def __gather_prediction_maps(models):
    """
    Gather prediction maps from the models and check for feature consistency.

    Input:
        models (MlModel): A variable number of models to compare.

    Returns:
        tuple: A tuple containing the list of prediction maps and a dictionary of model features.
    """
    prediction_maps = []
    model_features_dict = {}

    for model in models:
        key = f"{model.type}_{model.test_size}"
        if (
            key in model_features_dict
            and model_features_dict[key] != model.features_list
        ):
            warnings.warn(
                f"Models with type '{model.type}' and test size '{model.test_size}' "\
                    "have different features."
            )
        if model.prediction_map is None:
            raise AttributeError(
                f"Model {model.type} does not have a prediction_map attribute."
            )
        prediction_maps.append(model.prediction_map)
        model_features_dict[key] = model.features_list

    return prediction_maps, model_features_dict


def __find_map_bounds(prediction_maps):
    """
    Find the bounding coordinates for the map.

    Input:
        prediction_maps (list): List of prediction maps.

    Returns:
        tuple: A tuple containing the min and max latitude and longitude.
    """
    possible_long_columns = ["long", "x", "xcoord", "lon", "longitude"]
    possible_lat_columns = ["lat", "y", "ycoord", "latitude"]

    pred_map_columns = prediction_maps[0].columns
    idx_long_column = __get_column_indices(pred_map_columns, possible_long_columns)
    idx_lat_column = __get_column_indices(pred_map_columns, possible_lat_columns)

    if idx_long_column is None or idx_lat_column is None:
        raise ValueError("Longitude or latitude columns not found in prediction map.")

    xmin, xmax = 180, -180
    ymin, ymax = 90, -90

    for pred_map in prediction_maps:
        if pred_map.shape != prediction_maps[0].shape:
            warnings.warn("The models have different prediction_map structures.")
        max_long, min_long = (
            pred_map[idx_long_column].max(),
            pred_map[idx_long_column].min(),
        )
        max_lat, min_lat = (
            pred_map[idx_lat_column].max(),
            pred_map[idx_lat_column].min(),
        )
        xmax, xmin = max(xmax, max_long), min(xmin, min_long)
        ymax, ymin = max(ymax, max_lat), min(ymin, min_lat)

    return xmin, xmax, ymin, ymax


def __plot_predictions(
    prediction_maps, models, shp_filepath, filepath, xmin, xmax, ymin, ymax
):
    """
    Plot the predictions on a map.

    Input:
        prediction_maps (list): List of prediction maps.
        models (MlModel): List of models.
        shp_filepath (str): The filepath to the shapefile.
        filepath (str): The filepath to save the map to.
        xmin (float): Minimum longitude.
        xmax (float): Maximum longitude.
        ymin (float): Minimum latitude.
        ymax (float): Maximum latitude.

    Returns:
        None
    """
    shapefile = gpd.read_file(shp_filepath)
    color_map = ListedColormap(['#1f77b4', '#ff7f0e'])
    all_labels = pd.concat(prediction_maps)['label']
    unique_labels = sorted(all_labels.dropna().unique())
    offset = 0.2
    for i, pred_map in enumerate(prediction_maps):
        _, ax = plt.subplots()
        shapefile.plot(ax=ax, color='white', edgecolor='black')
        ax.set_xlim(xmin - offset, xmax + offset)
        ax.set_ylim(ymin - offset, ymax + offset)
        ax.set_title(f'Model {models[i].type}_{models[i].test_size} Predictions')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        idx_long_column = __get_column_indices(pred_map.columns, ["long", "x", "xcoord", "lon", "longitude"])
        idx_lat_column = __get_column_indices(pred_map.columns, ["lat", "y", "ycoord", "latitude"])
        if idx_long_column is None or idx_lat_column is None:
            raise ValueError("Longitude or latitude columns not found in prediction map.")
        for label in unique_labels: 
            label_data = pred_map[pred_map['label'] == label]
            ax.scatter(label_data[idx_long_column], label_data[idx_lat_column], 
                       c=[color_map(label)], label=f'Label {label}', alpha=0.5)
        ax.legend()
        if filepath is not None:
            plt.savefig(f"{filepath}_{models[i].type}_{models[i].test_size}.png")
        else:
            plt.show()

def plot_heatmap(model: MlModel, filepath: str = None):
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
        raise TypeError("Filepath must be a string.")
    plt.figure(figsize=(10, 8))
    numeric_data = model.dataset.select_dtypes(include=[float, int])
    if model.target_column in numeric_data.columns:
        numeric_data = numeric_data.drop(columns=[model.target_column])
    keywords = ["coord", "loc", "location", "coordinates", "pos", "position"]
    columns_to_exclude = [
        col
        for col in numeric_data.columns
        if any(keyword in col.lower() for keyword in keywords)
    ]
    numeric_data = numeric_data.drop(columns=columns_to_exclude)
    sns.heatmap(
        numeric_data.corr(),
        xticklabels=numeric_data.columns,
        yticklabels=numeric_data.columns,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
    )
    plt.title("Heatmap of Dataset Features")
    if filepath is not None:
        plt.savefig(filepath)
    elif filepath is None:
        plt.show()

def compare_metrics(
    *models: MlModel, filepath: str = None, palette: str = "dark:skyblue"
):
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
        raise TypeError("Filepath must be a string.")
    if len(models) < 2:
        raise ValueError("At least two models are required for comparison.")

    reference_features = models[0].features_list
    for i, model in enumerate(models[1:], start=2):
        if model.features_list != reference_features:
            warnings.warn(f"Model {i} has different features from the first model.")

    metrics_df = __create_metrics_df(models)
    model_types = [model.type for model in models]
    model_labels = models[0].dataset[models[0].target_column].unique()
    labels_list = [str(item) for item in model_labels]
    labels_list.append("accuracy")
    classes = labels_list
    metric_order = ["precision", "recall", "f1-score", "accuracy"]

    filtered_df = metrics_df[
        (metrics_df["Model"].isin(model_types)) & (metrics_df["Class"].isin(classes))
    ]
    pivot_df = filtered_df.pivot_table(
        index=["Model", "Class", "Metric"], values="Value"
    ).reset_index()

    _, axes = plt.subplots(nrows=len(metric_order), ncols=1, sharey="row")

    for i, metric in enumerate(metric_order):
        if metric != "accuracy":
            plot_data = pivot_df[pivot_df["Metric"] != "accuracy"]
        else:
            plot_data = pivot_df[pivot_df["Metric"] == "accuracy"]
        sns.barplot(
            x="Model",
            y="Value",
            hue="Class",
            data=plot_data[plot_data["Metric"] == metric],
            ax=axes[i],
            palette=palette,
        )
        axes[i].set_title(f"{metric.capitalize()} - Labels")
        axes[i].set_xlabel("")
        axes[i].set_ylabel(metric.capitalize())
        if metric != "accuracy":
            handles, labels = axes[i].get_legend_handles_labels()
            axes[i].legend(handles=handles[:2], labels=labels[:2])
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

def plot_map(*models: "MlModel", filepath: str = None, shp_filepath: str = None):
    """
    Plot the map of the dataset for each model in an overlay. The map shows the predicted label
    in a different color for each model. The map can be saved to a file if a filepath is provided.
    Otherwise, the map is displayed.

    Input:
        models (MlModel): A variable number of models to compare.
        filepath (str): The filepath to save the map to.
        shp_filepath (str): The filepath to the shapefile.

    Raises:
        TypeError: If the filepath is not a string.
        TypeError: If the models have different features.

    Returns:
        None
    """
    if not isinstance(filepath, str) and filepath is not None:
        raise TypeError("Filepath must be a string.")
    if shp_filepath is None:
        raise ValueError("Shapefile filepath must be provided.")
    if not isinstance(shp_filepath, str):
        raise TypeError("Shapefile filepath must be a string.")

    prediction_maps, _ = __gather_prediction_maps(models)
    xmin, xmax, ymin, ymax = __find_map_bounds(prediction_maps)

    __plot_predictions(
        prediction_maps, models, shp_filepath, filepath, xmin, xmax, ymin, ymax
    )
