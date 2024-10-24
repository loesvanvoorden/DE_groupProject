# Using python 3.8 as the base image
FROM python:3.8

# Defining working directory for the prediction API
WORKDIR /app 

# Copy requirements.txt to the working directory
COPY requirements.txt . 

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files in prediction-api-gcs local host directory to /app in Docker container
COPY . .  

# Copy the model file from lab2/models to /app in the Docker container
COPY ../models/model-train.pkl ./model-train.pkl

# Expose the port that our app runs on
EXPOSE 5000

# Environment Variables
ENV PROJECT_ID=de2024-436414
ENV MODEL_REPO=gs://assignment1de
ENV MODEL_NAME=model-train.pkl  

# Run our App
CMD ["python3", "app.py"]
