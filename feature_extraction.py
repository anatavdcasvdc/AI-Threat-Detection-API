import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

# Paths
processed_data_path = "D:/AI_Threat_Detection/dataset/processed"
output_data_path = "D:/AI_Threat_Detection/dataset/processed"

# Load preprocessed data
train_df = pd.read_csv(os.path.join(processed_data_path, 'train_final.csv'))
test_df = pd.read_csv(os.path.join(processed_data_path, 'test_final.csv'))

# Separate features and target
X_train = train_df.drop(columns=['class'])
y_train = train_df['class']
X_test = test_df.drop(columns=['class'])
y_test = test_df['class']

# 🚨 Drop columns with ONLY NaN values
X_train = X_train.dropna(axis=1, how='all')
X_test = X_test.dropna(axis=1, how='all')

# Handle missing values (Imputation)
imputer = SimpleImputer(strategy="mean")  # Replace NaNs with column mean
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Feature Selection: Select Top 20 Features
selector = SelectKBest(score_func=f_classif, k=20)
X_train_selected = selector.fit_transform(X_train_imputed, y_train)
X_test_selected = selector.transform(X_test_imputed)

# Get selected feature names
selected_features = X_train.columns[selector.get_support()]
X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)

# Add target column back
X_train_selected['class'] = y_train.values
X_test_selected['class'] = y_test.values

# Save extracted features
X_train_selected.to_csv(os.path.join(output_data_path, 'train_features.csv'), index=False)
X_test_selected.to_csv(os.path.join(output_data_path, 'test_features.csv'), index=False)

print("✅ Feature Extraction Completed! Saved selected features in dataset/processed/")
