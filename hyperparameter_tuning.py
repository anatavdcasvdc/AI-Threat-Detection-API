import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Paths
processed_data_path = "D:/AI_Threat_Detection/dataset/processed"

# Load data
train_df = pd.read_csv(os.path.join(processed_data_path, 'train_features.csv'))
test_df = pd.read_csv(os.path.join(processed_data_path, 'test_features.csv'))

# Separate features and target
X_train = train_df.drop(columns=['class'])
y_train = train_df['class']
X_test = test_df.drop(columns=['class'])
y_test = test_df['class']

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],   # Number of trees
    'max_depth': [10, 20, None],      # Tree depth
    'min_samples_split': [2, 5, 10]   # Min samples to split a node
}

# Perform Grid Search
clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from tuning
best_model = grid_search.best_estimator_
print("🚀 Best Parameters:", grid_search.best_params_)

# Evaluate on test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Tuned Model Accuracy: {accuracy:.4f}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save the tuned model
model_path = os.path.join(processed_data_path, "random_forest_tuned.pkl")
joblib.dump(best_model, model_path)
print(f"💾 Tuned model saved at: {model_path}")
