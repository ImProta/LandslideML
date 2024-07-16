"""
This script shows the workflow for generating a random forest model using the LandslideML library.

The script shows the main steps to generate a random forest model and predict with a NetCDF file:
1. Imports the library.
2. Defines the main variables for the model (data path, model type, target variable, feature list, 
    and test size).
3. Initializes the model with `generate_model` function.
4. Modifies the model parameters with the `setup` method.
5. Predicts the model with a NetCDF file.

The script also shows how to save the model, and load the model.

Note: Make sure to provide the correct data path, feature list, and test size before running.
the script.
"""

import landslideml as lsm

# Define variables for the model
DATA_PATH = "./testcase_data/training.csv"
MODEL_TYPE = "RandomForest"
FEATURE_LIST = ["alti",
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
                "n_mrc"]
TARGET = 'label'
TEST_SIZE = 0.25

# Initialize the model
random_forest = lsm.generate_model(DATA_PATH,
                                   MODEL_TYPE,
                                   FEATURE_LIST,
                                   TARGET,
                                   TEST_SIZE)

# Modify the model parameters
random_forest.setup(n_estimators=100, max_depth=15, random_state=42)

# Predict the model with a csv file
random_forest.predict("testcase_data/prediction_cropped.nc")

# Evaluate the model
random_forest.evaluate_model()

# Save the model
random_forest.save_model("random_forest_model.pkl")

# Load the model
loaded_model = lsm.load_model("random_forest_model.pkl")

# Verify the loaded model
print(loaded_model.prediction_map.head())
