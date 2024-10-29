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
import numpy as np

class PerformancePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        # Define base directory for the project
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Parent of current directory

    # Download the model from GCS
    def download_model(self):
<<<<<<< Updated upstream
        project_id = os.environ.get('PROJECT_ID', 'Specified environment variable is not set.')
        model_repo = os.environ.get('MODEL_REPO', 'Specified environment variable is not set.')
        model_name = os.environ.get('MODEL_NAME', 'Specified environment variable is not set.')
        client = storage.Client(project=project_id)
        bucket = client.bucket(model_repo)
        blob = bucket.blob(model_name)
        blob.download_to_filename('model_train_v1.pkl')
        self.model = joblib.load('model_train_v1.pkl')
=======
        try:
            project_id = os.environ.get('PROJECT_ID')
            model_repo = os.environ.get('MODEL_REPO')
            model_name = os.environ.get('MODEL_NAME')
>>>>>>> Stashed changes

            if not project_id or not model_repo or not model_name:
                raise ValueError("One or more required environment variables are not set.")

            client = storage.Client(project=project_id)
            bucket = client.bucket(model_repo)
            blob = bucket.blob(model_name)
            blob.download_to_filename('local_model.pkl')
            self.model = joblib.load('local_model.pkl')

        except Exception as e:
            logging.error(f"Failed to download the model: {e}")
            raise e
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

        # Convert the input JSON to a DataFrame
        df = pd.DataFrame(prediction_input)

        # Type casting to ensure compatibility
        df['schoolsup'] = df['schoolsup'].astype(str)
        df['higher'] = df['higher'].astype(str)

        # Casting numerical columns to float to ensure consistency
        numerical_columns = ['absences', 'failures', 'Medu', 'Fedu', 'Walc', 'Dalc', 'famrel', 'goout', 'freetime',
                             'studytime']
        for col in numerical_columns:
            df[col] = df[col].astype(float)

        # Check for NaN values
        if df.isnull().values.any():
            return jsonify({'error': 'Input data contains missing values'}), 400

        # Make prediction
        try:
            y_classes = self.model.predict(df[['schoolsup', 'higher', 'absences', 'failures', 'Medu', 'Fedu', 'Walc',
                                               'Dalc', 'famrel', 'goout', 'freetime', 'studytime']])
            logging.info(y_classes)
            df['pclass'] = np.where(y_classes > 0.5, 1, 0)
            status = df['pclass'][0]
            return jsonify({'predicted_class': int(status)}), 200
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return jsonify({'error': 'Prediction failed due to an internal error'}), 500


