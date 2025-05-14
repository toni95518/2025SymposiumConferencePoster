import pandas as pd

data = pd.read_csv('diabetes_prediction_dataset.csv')

# Remove rows with NaN/null or empty values
cleaned_data = data.dropna()

# Display summary again to verify the cleaning
print(cleaned_data.describe())