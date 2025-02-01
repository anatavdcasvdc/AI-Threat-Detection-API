import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
processed_data_path = "D:/AI_Threat_Detection/dataset/processed"
original_model_path = os.path.join(processed_data_path, "random_forest_model.pkl")
tuned_model_path = os.path.join(processed_data_path, "random_forest_tuned.pkl")

# Load test data
test_df = pd.read_csv(os.path.join(processed_data_path, 'test_features.csv'))
X_test = test_df.drop(columns=['class'])
y_test = test_df['class']

# Load both models
original_clf = joblib.load(original_model_path)
tuned_clf = joblib.load(tuned_model_path)
print("✅ Models loaded successfully!")

# Make predictions with both models
y_pred_original = original_clf.predict(X_test)
y_pred_tuned = tuned_clf.predict(X_test)

# Evaluate performance of both models
def evaluate_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n🚀 {model_name} Model Accuracy: {accuracy:.4f}")
    print(f"📊 {model_name} Classification Report:\n", classification_report(y_true, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

# Compare both models
evaluate_model("Original", y_test, y_pred_original)
evaluate_model("Tuned", y_test, y_pred_tuned)
