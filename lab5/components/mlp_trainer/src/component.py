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
from flask import jsonify

import json


from google.cloud import storage
import pandas as pd
import logging
import sys
import pickle
import os
import json
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from GCS to a local path."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logging.info(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")

def train_mlp(project_id, feature_path, model_repo, metrics_path):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Define local path for CSV
    local_csv_path = '/tmp/feature_data.csv'
    
    # Extract bucket and blob names from feature_path
    bucket_name = feature_path.split("/")[2]
    blob_name = "/".join(feature_path.split("/")[3:])

    # Download CSV from GCS
    download_from_gcs(bucket_name, blob_name, local_csv_path)
    
    # Load CSV into pandas
    df = pd.read_csv(local_csv_path, index_col=None)

    logging.info("Columns in the dataset: %s", df.columns)

    # Rest of the code for data processing and training...
    categorical_columns = ['schoolsup', 'higher']
    numerical_columns = ['absences', 'failures', 'Medu', 'Fedu', 'Walc', 'Dalc', 'famrel', 'goout', 'freetime', 'studytime']
    columns_to_average = ['G1', 'G2', 'G3']
    df['Average_Grade'] = df[columns_to_average].mean(axis=1)
    df['Average_Grade_Cat_1'] = pd.cut(df['Average_Grade'], bins=[0, 10, 20], labels=[0, 1], include_lowest=True)
    X = df[numerical_columns + categorical_columns]
    y = df['Average_Grade_Cat_1']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(drop='if_binary'), categorical_columns)
        ])

    svm_linear_clf = SVC(C=0.1, kernel='linear', gamma='scale', class_weight='balanced', degree=2, probability=True)
    svm_rbf_clf = SVC(C=0.1, kernel='rbf', gamma='scale', class_weight='balanced', degree=2, probability=True)
    logreg = LogisticRegression(C=10, solver='newton-cg')

    stack_clf = StackingClassifier(
        estimators=[
            ('svm_linear_clf', svm_linear_clf),
            ('svm_rbf', svm_rbf_clf)
        ],
        final_estimator=logreg,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', stack_clf)
    ])

    logging.info("Training the model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {accuracy}")
    print(f"Test Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    local_model_path = '/tmp/model_train.pkl'
    with open(local_model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    client = storage.Client(project=project_id)
    bucket = client.bucket(model_repo)
    blob = bucket.blob('model.pkl')
    blob.upload_from_filename(local_model_path)
    os.remove(local_model_path)
    logging.info(f"Model saved to GCP bucket: {model_repo}")

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as outfile:
        json.dump({"accuracy": accuracy}, outfile)


# Defining and parsing the command-line arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, help="GCP project id")
    parser.add_argument('--feature_path', type=str, help="CSV file with features")
    parser.add_argument('--model_repo', type=str, help="Name of the model bucket")
    parser.add_argument('--metrics_path', type=str, help="Name of the file to be used for saving evaluation metrics")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    train_mlp(**parse_command_line_arguments())