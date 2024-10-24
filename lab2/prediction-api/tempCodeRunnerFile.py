import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
from flask import jsonify
from io import StringIO
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class PerformancePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None

    def load_model(self, file_path):
        self.model = pickle.load(open(file_path, 'rb'))

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
        if self.model is None:
            try:
                model_repo = os.environ['MODEL_REPO']
                file_path = os.path.join(model_repo, "model_train.pkl")
                self.model = pickle.load(open(file_path, 'rb'))
            except KeyError:
                print("MODEL_REPO is undefined")
                self.model = pickle.load(open("model_train.pkl", 'rb'))

        # Convert prediction_input to DataFrame
        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')
        
        # If the preprocessor is not fitted, fit it with some dummy data or previous training data
        if self.preprocessor is None:
            # Load the training data to fit the preprocessor; replace this path with your actual training data path
            train_data_path = os.path.join('lab2', 'data', 'train_data.json')
            print(train_data_path)
            training_data = pd.read_json(train_data_path)
            self.fit_preprocessor(training_data)

        # Apply preprocessor to the new data
        xNew = self.preprocessor.transform(df[['schoolsup', 'higher', 'absences', 'failures', 'Medu', 'Fedu', 'Walc', 'Dalc', 'famrel', 'goout', 'freetime', 'studytime']])
        
        # Perform prediction
        y_classes = self.model.predict(xNew)
        logging.info(y_classes)

        # Assuming your model now predicts classes 0 and 1
        df['pclass'] = y_classes.tolist()  # Assign predicted classes (0 or 1)
        # Check the class prediction directly
        status = df['pclass'][0]  # status will be 0 or 1


        if status == 0:
            return "Predicted class: 0"
        elif status == 1:
            return "Predicted class: 1"


