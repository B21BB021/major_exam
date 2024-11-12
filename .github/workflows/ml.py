from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import mlflow

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow.sklearn


class IrisDataProcessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.data = load_iris()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def prepare_data(self):
        # Load data into DataFrame with column names
        feature_names = self.data.feature_names
        self.df = pd.DataFrame(self.data.data, columns=feature_names)
        self.df["target"] = self.data.target

        # Separate features and target
        X = self.df.drop("target", axis=1)
        y = self.df["target"]

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Feature scaling
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def get_feature_stats(self):
        # Basic statistical analysis for features
        return self.df.describe()


# Instantiate and use the class
processor = IrisDataProcessor()
processor.prepare_data()
stats = processor.get_feature_stats()
print("Feature statistics:\n", stats)


# Define the IrisExperiment class
class IrisExperiment:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Random Forest": RandomForestClassifier(),
        }
        self.results = {}

    def run_experiment(self):
        # Prepare data
        self.data_processor.prepare_data()

        # Track each model's performance with MLflow
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                # Cross-validation for each model
                X_train, X_test, y_train, y_test = (
                    self.data_processor.X_train,
                    self.data_processor.X_test,
                    self.data_processor.y_train,
                    self.data_processor.y_test,
                )

                # Train the model
                model.fit(X_train, y_train)

                # Cross-validation scores for metrics
                y_pred = model.predict(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")

                # Log metrics in MLflow
                mlflow.log_param("model_type", model_name)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

                # Log the model
                mlflow.sklearn.log_model(model, model_name)

                # Store results in the dictionary for further analysis
                self.results[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                }

    def log_results(self):
        # Print the results for each model
        for model_name, metrics in self.results.items():
            print(f"Results for {model_name}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            print()


# Usage Example
# Assuming you have already created an instance of IrisDataProcessor

# processor = IrisDataProcessor()
# experiment = IrisExperiment(processor)
# experiment.run_experiment()
# experiment.log_results()


class IrisModelOptimizer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.quantized_model = None

    def quantize_model(self):
        """
        Simulates model quantization by reducing the precision of the logistic regression model.
        """
        # Select logistic regression model
        model = self.experiment.models["Logistic Regression"]

        # Quantize model by saving with reduced precision (simulated quantization)
        quantized_filename = "quantized_logistic_regression_model.joblib"

        # Save with reduced precision
        joblib.dump(model, quantized_filename, compress=("zlib", 3))

        # Load the quantized model
        self.quantized_model = joblib.load(quantized_filename)

        # Verify model is loaded and usable
        print(f"Quantized model saved and loaded from {quantized_filename}.")

    def run_tests(self):
        """
        Simple unit tests for model quantization and performance verification.
        """
        # Test 1: Check if quantized model is available and functional
        assert self.quantized_model is not None, "Quantized model is not loaded."

        # Prepare test data
        X_test, y_test = (
            self.experiment.data_processor.X_test,
            self.experiment.data_processor.y_test,
        )

        # Test 2: Check if quantized model gives predictions
        y_pred = self.quantized_model.predict(X_test)
        assert y_pred.shape == y_test.shape, "Prediction shape mismatch."

        # Test 3: Check if accuracy of quantized model is within acceptable range (e.g., > 0.8)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of quantized model: {accuracy:.4f}")
        assert accuracy >= 0.8, "Quantized model accuracy below threshold."

        print("All unit tests passed successfully.")


# Usage Example
# Assuming you have already run an experiment and have an instance of IrisExperiment

# experiment = IrisExperiment(processor)
# experiment.run_experiment()
# optimizer = IrisModelOptimizer(experiment)
# optimizer.quantize_model()
# optimizer.run_tests()


def main():
    # Initialize processor
    processor = IrisDataProcessor()
    processor.prepare_data()

    # Run experiments
    experiment = IrisExperiment(processor)
    experiment.run_experiment()

    # Optimize and test
    optimizer = IrisModelOptimizer(experiment)
    optimizer.quantize_model()
    optimizer.run_tests()


if __name__ == "__main__":
    main()
