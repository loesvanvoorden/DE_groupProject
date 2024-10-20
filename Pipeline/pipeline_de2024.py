# Create preprocessing component
@component(base_image="gcr.io/de2024-436414/pre-processing:v1")
def preprocess_component(user_data: str, output_gcs_path: str):
    # Simulate preprocessing task (replace with actual preprocessing logic)
    print(f"Preprocessing user data: {user_data}")
    # Simulate saving preprocessed data to GCS
    preprocessed_data_path = output_gcs_path
    print(f"Preprocessed data saved to {preprocessed_data_path}")
    return preprocessed_data_path

# Create training component
@component(base_image="gcr.io/de2024-436414/training-testing:v1")
def train_component(input_data_path: str, model_output_path: str):
    # Simulate training task (replace with actual model training logic)
    print(f"Training model using data: {input_data_path}")
    # Simulate saving trained model to GCS
    trained_model_path = model_output_path
    print(f"Model saved to {trained_model_path}")
    return trained_model_path

# Create prediction component
@component(base_image="gcr.io/de2024-436414/prediction:v1")
def prediction_component(preprocessed_data_path: str, model_path: str, output_prediction_path: str):
    # Simulate prediction task (replace with actual prediction logic)
    print(f"Making predictions using preprocessed data: {preprocessed_data_path} and model: {model_path}")
    # Simulate saving predictions to GCS
    predictions_path = output_prediction_path
    print(f"Predictions saved to {predictions_path}")
    return predictions_path

# Define the pipeline
@pipeline(name="de2024-436414-pipeline")
def my_pipeline(user_data: str):
    # Define GCS paths
    output_gcs_path = 'gs://your-bucket/output/preprocessed.csv'
    model_output_path = 'gs://your-bucket/output/model.pkl'
    prediction_output_path = 'gs://your-bucket/output/predictions.csv'
    
    # Pipeline tasks
    preprocess_task = preprocess_component(user_data=user_data, output_gcs_path=output_gcs_path)
    train_task = train_component(input_data_path=preprocess_task.outputs['output_gcs_path'], model_output_path=model_output_path)
    predict_task = prediction_component(preprocessed_data_path=preprocess_task.outputs['output_gcs_path'], model_path=train_task.outputs['model_output_path'], output_prediction_path=prediction_output_path)

# Compile the pipeline
from kfp.v2 import compiler
compiler.Compiler().compile(pipeline_func=my_pipeline, package_path='pipeline_de2024.json')

# Submit the pipeline to Vertex AI
def submit_pipeline():
    aiplatform.init(project='de2024-436414', location='us-central1')

    job = aiplatform.PipelineJob(
        display_name="de2024-436414-pipeline",
        template_path='pipeline_de2024.json',
        parameter_values={'user_data': '{"absences": 3, "failures": 1, "study_time": 10, "parental_education": 2}'}, # Example synthetic data
        pipeline_root='gs://your-bucket-name/pipeline-root'
    )
    job.run()

# Entry point for running the script
if __name__ == "__main__":
    submit_pipeline()
