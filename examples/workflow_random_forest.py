"""
This script shows the workflow for generating a random forest model using the LandslideML library.

The script shows the 4 mains steps to generate a model:
1. Imports the library.
2. Defines the main variables for the model (data path, model type, target variable, and 
 feature list).
3. Initializes the model with `generate_model` function.
4. Modifies the model parameters with the `setup` method.

Example usage:
python workflow_random_forest.py

Note: Make sure to provide the correct data path and feature list before running the script.
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

# Predict the model
# random_forest.predict(filepath="testcase_data/prediction.nc")

# Evaluate the model
# random_forest.evaluate_model()

# Save the model
# random_forest.save_model("random_forest_model.pkl")
