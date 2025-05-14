import pandas as pd
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(file_path):
    logging.info("Loading and cleaning data")
    data = pd.read_csv(file_path)
    data = data.dropna()
    valid_data = data[
        (data['age'] >= 0) & (data['age'] <= 120) &
        (data['bmi'] >= 10) & (data['bmi'] <= 60) &
        (data['HbA1c_level'] >= 2.5) & (data['HbA1c_level'] <= 15) &
        (data['blood_glucose_level'] >= 50) & (data['blood_glucose_level'] <= 500)
    ]
    logging.info("Data loaded and cleaned")
    return valid_data

def preprocess_data(data, categorical_features):
    logging.info("Preprocessing data")
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = pd.DataFrame(encoder.fit_transform(data[categorical_features]))
    encoded_features.columns = encoder.get_feature_names_out(categorical_features)
    data = data.drop(columns=categorical_features)
    data = pd.concat([data.reset_index(drop=True), encoded_features.reset_index(drop=True)], axis=1)
    scaler = MinMaxScaler()
    numerical_features = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    logging.info("Data preprocessed")
    return data

def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating model")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    logging.info("Model evaluated")
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity
    }

def main():
    try:
        file_path = 'diabetes_prediction_dataset.csv'
        categorical_features = ['gender', 'smoking_history']
        data = load_and_clean_data(file_path)
        data = preprocess_data(data, categorical_features)
        X = data.drop(columns=['diabetes'])
        y = data['diabetes']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize models with reduced parameter grids for faster grid search
        model_params = {
            "Random Forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100],  # Reduced parameter grid
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                }
            },
            "Decision Tree": {
                "model": DecisionTreeClassifier(random_state=42),
                "params": {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                }
            },
            "SVM": {
                "model": SVC(random_state=42),
                "params": {
                    "C": [0.1, 1, 10],
                    "gamma": [0.1, 0.01],
                    "kernel": ["rbf"]
                }
            }
        }

        metrics = {}

        for model_name, mp in model_params.items():
            logging.info(f"Starting grid search for {model_name}")
            grid_search = GridSearchCV(mp["model"], mp["params"], cv=3, scoring='accuracy', n_jobs=-1)  # Use n_jobs=-1 for parallel processing
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            metrics[model_name] = evaluate_model(best_model, X_test, y_test)
            logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

        metrics_df = pd.DataFrame(metrics).T
        print(metrics_df)
        logging.info("Script completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()