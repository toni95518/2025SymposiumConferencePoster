import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def load_and_clean_data(file_path):
    """
    Load the dataset and clean it by removing NaN values and invalid entries.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise Exception(f"File {file_path} not found.")
    
    # Remove rows with NaN/null values
    data = data.dropna()

    # Remove rows with invalid values based on given criteria
    valid_data = data[
        (data['age'] >= 0) & (data['age'] <= 120) &
        (data['bmi'] >= 10) & (data['bmi'] <= 60) &
        (data['HbA1c_level'] >= 2.5) & (data['HbA1c_level'] <= 15) &
        (data['blood_glucose_level'] >= 50) & (data['blood_glucose_level'] <= 500)
    ]

    return valid_data

def preprocess_data(data, categorical_features):
    """
    Preprocess the data by encoding categorical features and normalizing numerical features.
    """
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = pd.DataFrame(encoder.fit_transform(data[categorical_features]))

    # Ensure the encoded feature names are preserved
    encoded_features.columns = encoder.get_feature_names_out(categorical_features)

    # Concatenate encoded features with the rest of the data
    data = data.drop(columns=categorical_features)
    data = pd.concat([data.reset_index(drop=True), encoded_features.reset_index(drop=True)], axis=1)

    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_features = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance using various metrics.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity
    }

def main():
    file_path = 'diabetes_prediction_dataset.csv'
    categorical_features = ['gender', 'smoking_history']
    
    # Load and clean data
    data = load_and_clean_data(file_path)

    # Preprocess data
    data = preprocess_data(data, categorical_features)
    
    # Split data into features and target
    X = data.drop(columns=['diabetes'])
    y = data['diabetes']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(random_state=42)
    }

    # Dictionary to store evaluation metrics for each model
    metrics = {}

    # Train and evaluate each model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        metrics[model_name] = evaluate_model(model, X_test, y_test)

    # Convert metrics to DataFrame for better visualization
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df)

if __name__ == "__main__":
    main()