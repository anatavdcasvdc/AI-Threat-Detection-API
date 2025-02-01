import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
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

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save Model
model_path = os.path.join(processed_data_path, "random_forest_model.pkl")
joblib.dump(clf, model_path)
print(f"💾 Model saved at: {model_path}")
