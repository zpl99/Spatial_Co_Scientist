import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer
import os

# Load the dataset
train_path = "benchmark/datasets/dkpes/dkpes_train.csv"
test_path = "benchmark/datasets/dkpes/dkpes_test.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Define the threshold for binary classification
threshold = 0.5  # Signal inhibition values >= 0.5 are labeled as 1, otherwise 0

# Prepare training data
X_train = train_data.drop(columns=["index", "Signal-inhibition"])
y_train = Binarizer(threshold=threshold).fit_transform(train_data[["Signal-inhibition"]]).ravel()

# Prepare test data
X_test = test_data.drop(columns=["index", "Signal-inhibition"])
test_indices = test_data["index"]

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
test_predictions = rf_classifier.predict(X_test)

# Save predictions to CSV
output_dir = "pred_results"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "dkpes_test_pred.csv")
pred_results = pd.DataFrame({"index": test_indices, "Predicted-Signal-Inhibition": test_predictions})
pred_results.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")