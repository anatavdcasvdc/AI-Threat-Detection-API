import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Paths
processed_data_path = "D:/AI_Threat_Detection/dataset/processed"
output_data_path = "D:/AI_Threat_Detection/dataset/processed"

# Load datasets
train_df = pd.read_csv(os.path.join(processed_data_path, 'train_processed.csv'))
test_df = pd.read_csv(os.path.join(processed_data_path, 'test_processed.csv'))

# Convert byte-strings to regular strings
train_df = train_df.apply(lambda x: x.map(lambda y: y.decode('utf-8') if isinstance(y, bytes) else y))
test_df = test_df.apply(lambda x: x.map(lambda y: y.decode('utf-8') if isinstance(y, bytes) else y))

# Identify categorical and numerical columns
categorical_cols = ['protocol_type', 'service', 'flag']  # Categorical features
label_col = 'class'  # Target variable
numerical_cols = [col for col in train_df.columns if col not in categorical_cols + [label_col]]

# Convert categorical data to numeric using Label Encoding
encoder = LabelEncoder()
for col in categorical_cols:
    train_df[col] = encoder.fit_transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])

# Convert any remaining non-numeric data
train_df[numerical_cols] = train_df[numerical_cols].apply(pd.to_numeric, errors='coerce')
test_df[numerical_cols] = test_df[numerical_cols].apply(pd.to_numeric, errors='coerce')

# Normalize numerical columns using MinMaxScaler
scaler = MinMaxScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

# Save preprocessed data
train_df.to_csv(os.path.join(output_data_path, 'train_final.csv'), index=False)
test_df.to_csv(os.path.join(output_data_path, 'test_final.csv'), index=False)

print("✅ Data Preprocessing Completed! Processed files saved in dataset/processed/")
