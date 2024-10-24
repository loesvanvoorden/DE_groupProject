import os
import json
import logging
from io import StringIO
import pandas as pd
from flask import jsonify
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from google.cloud import storage
import joblib

class PerformancePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        # Define base directory for the project
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Parent of current directory

    # Download the model from GCS
    def download_model(self):
        project_id = os.environ.get('PROJECT_ID', 'Specified environment variable is not set.')
        model_repo = os.environ.get('MODEL_REPO', 'Specified environment variable is not set.')
        model_name = os.environ.get('MODEL_NAME', 'Specified environment variable is not set.')
        client = storage.Client(project=project_id)
        bucket = client.bucket(model_repo)
        blob = bucket.blob(model_name)
        blob.download_to_filename('local_model.pkl')
        self.model = joblib.load('local_model.pkl')

    def fit_preprocessor(self, dataset):
        categorical_columns = ['schoolsup', 'higher']
        numerical_columns = ['absences', 'failures', 'Medu', 'Fedu', 'Walc', 'Dalc', 'famrel', 'goout', 'freetime', 'studytime']
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),
                ('cat', OneHotEncoder(drop='if_binary'), categorical_columns)
            ]
        )
        self.preprocessor.fit(dataset[numerical_columns + categorical_columns])

    def predict_single_record(self, prediction_input):
        logging.debug(prediction_input)
        if self.model is None:
            self.download_model()

        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')

        if self.preprocessor is None:
            try:
                # Construct the path relative to the base directory
                train_data_path = os.path.join(self.base_dir, 'lab2', 'data', 'train_data.json')
                training_data = pd.read_json(train_data_path)
                self.fit_preprocessor(training_data)
            except FileNotFoundError:
                logging.error("Training data file not found.")
                return jsonify({'error': 'Training data file not found.'}), 500

        xNew = self.preprocessor.transform(df[['schoolsup', 'higher', 'absences', 'failures', 'Medu', 'Fedu', 'Walc', 'Dalc', 'famrel', 'goout', 'freetime', 'studytime']])
        y_classes = self.model.predict(xNew)
        logging.info(y_classes)

        df['pclass'] = np.where(y_classes > 0.5, 1, 0)
        status = df['pclass'][0]

        return jsonify({'predicted_class': int(status)}), 200
