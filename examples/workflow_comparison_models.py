import landslideml as lsm

DATA_PATH = "./testcase_data/training.csv"
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
                "saturated_water_content",
                "alpha_mrc",
                "n_mrc"]
TARGET = 'label'
TEST_SIZE = 0.20
FILEPATH_PREDICTION = "testcase_data/prediction.nc"

rf = lsm.generate_model(DATA_PATH,
                        "RandomForest",
                        FEATURE_LIST,
                        TARGET,
                        TEST_SIZE)

gbm = lsm.generate_model(DATA_PATH,
                        "GBM",
                        FEATURE_LIST,
                        TARGET,
                        TEST_SIZE)

svm = lsm.generate_model(DATA_PATH,
                        "SVM",
                        FEATURE_LIST,
                        TARGET,
                        TEST_SIZE)

rf.setup(n_estimators=250,
         max_features=3,
         min_samples_leaf=5,
         n_jobs=4,
         random_state=42
         )
svm.setup(C=100.0, 
          kernel='poly',
          degree=41,
          gamma='scale',
          random_state=42
          )
gbm.setup(loss='exponential',
          learning_rate=0.1,
          n_estimators=300,
          max_depth=5,
          random_state=42
          )

rf.evaluate_model(show=True)
svm.evaluate_model(show=True)
gbm.evaluate_model(show=True)

rf.predict(FILEPATH_PREDICTION)
svm.predict(FILEPATH_PREDICTION)
gbm.predict(FILEPATH_PREDICTION)

lsm.compare_metrics(rf, svm, gbm, filepath="run1_metrics_comparison")

lsm.plot_map(rf, svm, gbm, filepath="run1_map_comparison", shp_filepath="testcase_data/shapefile.shp")