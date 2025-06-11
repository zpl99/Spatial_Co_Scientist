import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# Load the datasets
train_data = pd.read_csv('benchmark/datasets/dkpes/dkpes_train.csv')
test_data = pd.read_csv('benchmark/datasets/dkpes/dkpes_test.csv')

# Define the threshold for binary classification
threshold = 0.5

# Convert signal inhibition values to binary labels
train_data['BinaryLabel'] = (train_data['Signal-inhibition'] > threshold).astype(int)

# Separate features and target variable
X_train = train_data.drop(columns=['index', 'Signal-inhibition', 'BinaryLabel'])
y_train = train_data['BinaryLabel']

X_test = test_data.drop(columns=['index', 'Signal-inhibition'])

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
test_predictions = rf_classifier.predict(X_test_scaled)

# Prepare the results DataFrame
results_df = pd.DataFrame({
    'index': test_data['index'],
    'PredictedSignalInhibition': test_predictions
})

# Ensure the output directory exists
output_dir = 'pred_results'
os.makedirs(output_dir, exist_ok=True)

# Save the predictions to a CSV file
results_df.to_csv(f'{output_dir}/dkpes_test_pred.csv', index=False)

print("Predictions saved successfully.")