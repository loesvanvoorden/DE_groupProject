# Using python 3.10 slim image as the base image
FROM python:3.10.7-slim
# Define working directory and copy the requirments file
WORKDIR /usr/src/pyapp
# Copy requirements.txt to the working directory
COPY requirements.txt .
# Install required python packages
RUN pip install --no-cache-dir -r requirements.txt
# Copy all files in DE_groupProject/prediction-ui host directory to /usr/src/myapp in Docker container
COPY . .
# Expose the port that our app runs in
EXPOSE 5000
# Environment Variables
ENV PREDICTOR_API=http://predictor-api-server:5000/prediction
#Add some meta data if necessary
LABEL diabetes.predictor.api="0.0.1-beta"
# Run our App
CMD ["python3","app.py"]
