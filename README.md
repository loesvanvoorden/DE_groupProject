# DE_groupProject

---

# MLOps System for Student Performance Prediction

### Data Engineering Assignment 1
**Roeland Kramer, Roos Mast, Loes van Voorden, and Levi Warren**  
**Date: October 30, 2024**

---

## Table of Contents

1. [Overview of the ML Application](#overview-of-the-ml-application)
2. [Goals and Requirements](#goals-and-requirements)
3. [System Design and Architecture](#system-design-and-architecture)
4. [Setup and Installation](#setup-and-installation)
5. [Pipeline Workflow](#pipeline-workflow)
6. [Reflection and Improvements](#reflection-and-improvements)
7. [Contributors](#contributors)

---

## Overview of the ML Application

This project presents an ML application designed to **predict student performance** based on demographic and academic factors. Leveraging a dataset from secondary school students, the application forecasts final grades as either "Fail" or "Succeed." The prediction model aims to help educators and institutions **identify at-risk students**, enabling timely interventions to improve academic outcomes.

**Dataset Overview**  
- **Columns**: 33 attributes including student demographics, family background, and academic history.
- **Purpose**: To analyze and predict the likelihood of student success or failure.

**Model**  
- **Type**: Stacking ensemble of Support Vector Machine models.
- **Source**: Adapted from Levi’s Introduction to Machine Learning project, where it achieved the highest accuracy among tested models.

## Goals and Requirements

### ML Application Goals
- **Predict student performance** accurately based on available features.
- **Assist educators** by identifying students needing additional support.
- **Enable timely interventions** to improve student outcomes.

### Technical and MLOps Requirements
- **Data Quality**: Ensure clean, consistent data by preprocessing missing values and handling outliers.
- **Model Performance**: High accuracy and generalization for reliable predictions.
- **User Accessibility**: A user-friendly interface for educators to input data easily and obtain predictions.
- **Compliance**: Anonymize sensitive data (e.g., student IDs) to ensure GDPR compliance.

### MLOps Requirements
- **Retraining**: Allow periodic model retraining to adapt to new data.
- **Monitoring**: Track model performance and detect any degradation, such as data drift.
- **API and UI**: Accessible endpoints for easy data input and output, without requiring technical expertise.

## System Design and Architecture

The MLOps system consists of three main components:

1. **ML Pipeline**  
   - **Ingestion and Preparation**: Data is imported from Google Cloud Storage, split into training and test sets, then processed for modeling.
   - **Training**: The stacked model is trained using a split dataset and saved as an artifact.
   - **Evaluation**: Model metrics are calculated, stored, and assessed for performance tracking.
   
2. **Prediction and Serving**  
   - **Prediction API**: Back-end endpoint for processing single-instance predictions.
   - **User Interface**: Front-end for educators to input data and view results.
   
3. **CI/CD Pipeline**  
   - **Automation**: Google Cloud Build handles building, testing, and deploying model components.
   - **Conditional Deployment**: Triggers determine if the model should be uploaded and deployed based on changes to data or configurations.

**See Figure 1 in the Appendix for a visual of the pipeline architecture.**

## Pipeline Workflow

1. **Data Ingestion**: Imports data and stores it as an artifact.
2. **Model Training**: Processes and trains a stacked model using a split dataset.
3. **Evaluation**: Computes metrics and logs results for monitoring.
4. **Prediction and Serving**: Accessible UI and API for generating predictions on new data: https://prediction-ui-925934865787.us-central1.run.app/checkperformance
5. **CI/CD Automation**: Automatically triggers model retraining or updates on code changes or data refreshes.

## Reflection and Improvements

**Alternative Design Considerations**  
- **Serverless Predictions**: Using Google Cloud Functions could simplify scaling and operational overhead.
- **Batch Prediction**: Implementing a batch processing system could optimize resource usage for large datasets.

**Potential Improvements**  
- **Real-time Model Monitoring**: Adding monitoring to track data drift and prediction accuracy.
- **Automated Hyperparameter Tuning**: Optimizing model accuracy by searching for the best parameter configurations.
- **Data Versioning**: Adding version control for training data to track and reproduce previous model versions.

## Contributors

- **Roeland Kramer** - CI/CD setup, UI/API configuration
- **Roos Mast** - Report writing, documentation
- **Loes van Voorden** - UI and API setup, Vertex AI configuration
- **Levi Warren** - Vertex AI pipeline, model training, and orchestration

**Acknowledgments**  
Special thanks to our instructors and mentors who guided us through the intricacies of MLOps implementation.

