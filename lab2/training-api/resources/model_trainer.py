import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from flask import jsonify

# Training and Testing Component
def train(dataset):
    # Define feature set X and target variable y
    categorical_columns = ['schoolsup', 'higher']
    numerical_columns = ['absences', 'failures', 'Medu', 'Fedu', 'Walc', 'Dalc', 'famrel', 'goout', 'freetime', 'studytime']
    X = dataset[numerical_columns + categorical_columns]
    y = dataset['Average_Grade_Cat_1']

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

    # Save the model as a .pkl file
    model_repo = os.environ.get('MODEL_REPO', None)
    if model_repo:
        model_output_path = os.path.join(model_repo, "model.pkl")  # Save to the repo as .pkl
    else:
        model_output_path = "model_train.pkl"  # default local path

    with open(model_output_path, 'wb') as f:
        pickle.dump(pipeline, f)

    logging.info(f"Model saved to {model_output_path}")
    print(f"Model saved to {model_output_path}")

    # Return JSON response
    if model_repo:
        return jsonify({'message': f'The model was saved to {model_repo}.'}), 200
    else:
        return jsonify({'message': 'The model was saved locally.'}), 200
