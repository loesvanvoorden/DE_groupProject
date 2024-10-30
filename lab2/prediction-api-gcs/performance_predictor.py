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

        # Convert the input JSON to a DataFrame
        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')

        # No need to manually handle the preprocessor as it's part of the model pipeline
        y_classes = self.model.predict(df[['schoolsup', 'higher', 'absences', 'failures', 'Medu', 'Fedu', 'Walc', 'Dalc', 'famrel', 'goout', 'freetime', 'studytime']])
        logging.info(y_classes)

        df['pclass'] = np.where(y_classes > 0.5, 1, 0)
        status = df['pclass'][0]

        return jsonify({"status": int(status)}), 200

