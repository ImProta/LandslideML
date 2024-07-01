from landslideml import DataPreprocessor, ModelTrainer, ModelEvaluator, Visualizer

# Example usage of the greet function
data_preprocessor = DataPreprocessor(filepath='path/to/dataset.csv')
data_preprocessor.greet("Victor")

# Load and preprocess data
data = data_preprocessor.load_data()
data_preprocessor.preprocess_data()

# Assume 'features' and 'target' are defined after preprocessing
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

# Train model
model_trainer = ModelTrainer(model_type='RandomForest')
model, X_test, y_test = model_trainer.train(X, y)

# Evaluate model
accuracy, report = ModelEvaluator.evaluate(model, X_test, y_test)
print(f'Accuracy: {accuracy}')
print(report)

# Visualize results
Visualizer.plot_results(y_test, model.predict(X_test))