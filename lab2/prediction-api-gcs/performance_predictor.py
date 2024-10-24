import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
from flask import jsonify
from google.cloud import storage
from keras.models import load_model
from io import StringIO
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class PerformancePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None

    # Download the model from GCS
    def download_model(self):
        project_id = os.environ.get('PROJECT_ID', 'Specified environment variable is not set.')
        model_repo = os.environ.get('MODEL_REPO', 'Specified environment variable is not set.')
        model_name = os.environ.get('MODEL_NAME', 'Specified environment variable is not set.')
        client = storage.Client(project=project_id)
        bucket = client.bucket(model_repo)
        blob = bucket.blob(model_name)
        blob.download_to_filename('local_model.pkl')
        self.model = load_model('local_model.pkl')

    def fit_preprocessor(self, dataset):
        # Define feature set and target variable for the preprocessor
        categorical_columns = ['schoolsup', 'higher']
        numerical_columns = ['absences', 'failures', 'Medu', 'Fedu', 'Walc', 'Dalc', 'famrel', 'goout', 'freetime', 'studytime']

        # Create the preprocessor using the same configurations as in the training
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),
                ('cat', OneHotEncoder(drop='if_binary'), categorical_columns)
            ]
        )
        # Fit the preprocessor on the dataset
        self.preprocessor.fit(dataset[numerical_columns + categorical_columns])

    def predict_single_record(self, prediction_input):
        logging.debug(prediction_input)
        # Download the model if it hasn't been loaded yet
        if self.model is None:
            self.download_model()

        # Convert prediction_input to DataFrame
        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')
        
        # If the preprocessor is not fitted, fit it with some dummy data or previous training data
        if self.preprocessor is None:
            try:
                train_data_path = os.path.join('lab2', 'data', 'train_data.json')
                training_data = pd.read_json(train_data_path)
                self.fit_preprocessor(training_data)
            except FileNotFoundError:
                logging.error("Training data file not found.")
                return jsonify({'error': 'Training data file not found.'}), 500

        # Apply preprocessor to the new data
        xNew = self.preprocessor.transform(df[['schoolsup', 'higher', 'absences', 'failures', 'Medu', 'Fedu', 'Walc', 'Dalc', 'famrel', 'goout', 'freetime', 'studytime']])
        
        # Perform prediction
        y_classes = self.model.predict(xNew)
        logging.info(y_classes)

        # Assuming your model now predicts classes 0 and 2
        df['pclass'] = np.where(y_classes > 0.5, 1, 0)  # Convert predictions to classes 0 and 2
        status = df['pclass'][0]  # Get the predicted class for the first record

        # Return the prediction outcome as a JSON message
        return jsonify({'predicted_class': int(status)}), 200
