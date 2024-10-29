import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from google.cloud import storage
import json


def train_mlp(project_id, feature_path, model_repo, metrics_path):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Load the dataset
    df = pd.read_csv(feature_path, index_col=None)
    logging.info(df.columns)

    # Define feature set X and target variable y
    categorical_columns = ['schoolsup', 'higher']
    numerical_columns = ['absences', 'failures', 'Medu', 'Fedu', 'Walc', 'Dalc', 'famrel', 'goout', 'freetime', 'studytime']
    columns_to_average = ['G1', 'G2', 'G3']
    df['Average_Grade'] = df[columns_to_average].mean(axis=1)
    df['Average_Grade_Cat_1'] = pd.cut(df['Average_Grade'], bins=[0, 10, 20], labels=[0, 1], include_lowest=True)
    X = df[numerical_columns + categorical_columns]
    y = df['Average_Grade_Cat_1']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define pre-processing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(drop='if_binary'), categorical_columns)
        ])

    # Define the base models
    svm_linear_clf = SVC(C=0.1, kernel='linear', gamma='scale', class_weight='balanced', degree=2, probability=True)
    svm_rbf_clf = SVC(C=0.1, kernel='rbf', gamma='scale', class_weight='balanced', degree=2, probability=True)

    # Meta-classifier
    logreg = LogisticRegression(C=10, solver='newton-cg')

    # Stacking ensemble classifier configuration
    stack_clf = StackingClassifier(
        estimators=[
            ('svm_linear_clf', svm_linear_clf),
            ('svm_rbf', svm_rbf_clf)
        ],
        final_estimator=logreg,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', stack_clf)
    ])

    # Fit the pipeline to the training data
    logging.info("Training the model with the following configuration...")
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {accuracy}")
    print(f"Test Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    # Save the model as a .pkl file in a local path
    local_file = '/tmp/model_train_v1.pkl'  # Local path to save the model
    with open(local_file, 'wb') as f:
        pickle.dump(pipeline, f)

    logging.info(f"Model saved to {local_file}")

    # Save to GCS as model.pkl
    client = storage.Client(project=project_id)
    bucket = client.get_bucket(model_repo)
    blob = bucket.blob('model.pkl')
    blob.upload_from_filename(local_file)  # Upload the locally saved model
    os.remove(local_file)  # Clean up
    logging.info("Saved the model to GCP bucket : " + model_repo)

    # Creating the directory where the output file is created (the directory
    # may or may not exist).
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as outfile:
        json.dump({"accuracy": accuracy}, outfile)


# Defining and parsing the command-line arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, help="GCP project id")
    parser.add_argument('--feature_path', type=str, help="CSV file with features")
    parser.add_argument('--model_repo', type=str, required=True, help="Name of the model bucket")
    parser.add_argument('--metrics_path', type=str, help="Name of the file to be used for saving evaluation metrics")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    # Ensure that only command-line arguments are used for model_repo
    train_mlp(**parse_command_line_arguments())