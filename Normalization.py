import pandas as pd

data = pd.read_csv('diabetes_prediction_dataset.csv')

# Remove rows with NaN/null or empty values
cleaned_data = data.dropna()

# Additional step to remove invalid values if necessary
cleaned_data = cleaned_data[
    (cleaned_data['age'] >= 0) & (cleaned_data['age'] <= 120) &
    (cleaned_data['bmi'] >= 10) & (cleaned_data['bmi'] <= 60) &
    (cleaned_data['HbA1c_level'] >= 2.5) & (cleaned_data['HbA1c_level'] <= 15) &
    (cleaned_data['blood_glucose_level'] >= 50) & (cleaned_data['blood_glucose_level'] <= 500)
]

# Min-Max normalization function
def min_max_normalize(df):
    result = df.copy()  # Create a copy of the dataframe
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:  # Apply normalization only to numeric columns
            min_value = df[column].min()
            max_value = df[column].max()
            result[column] = (df[column] - min_value) / (max_value - min_value)
    return result

# Apply Min-Max normalization
normalized_data = min_max_normalize(cleaned_data)

# Display summary of the normalized data to verify
print(normalized_data.describe())

# Save the normalized data to a new CSV file
normalized_data.to_csv('normalized_diabetes_prediction_dataset.csv', index=False)