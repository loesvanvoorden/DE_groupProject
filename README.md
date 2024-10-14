# DE_groupProject
Group project by Finn Franken, Ries Houthuijzen, Roeland Kramer, Roos Mast, Loes van Voorden and Levi Warren for the course Data Engineering

## Planning:
1. Select the Machine Learning Application
Goal: Pick an ML use case. You can reuse any past projects (as long as they are not already dockerized or set up with Vertex AI pipelines) or find relevant ML projects on GitHub/Kaggle.
Example: Past projects include breast cancer prediction​(Assignment 1 Example fr…)and toxicity classification​(Assignment 2 Example fr…).
Platform: Any Python-based machine learning codebase (from GitHub/Kaggle or your previous projects).

2. Set Up Google Cloud Platform (GCP)
Goal: Configure your working environment on Google Cloud.
Platform:
Google Cloud Platform (GCP): Set up your Virtual Machine (VM) and other services like Google Cloud Storage and Vertex AI.
**Key Actions**:
- Create and configure a VM to run your ML pipelines.
- Ensure Docker is installed on the VM, as you will need to containerize your models and services.
- Use Google Cloud Storage as the data source for your pipelines.

3. Design the ML Pipeline
Goal: Create an automated ML pipeline using Google Vertex AI. The pipeline includes data ingestion, training, and model validation.
Platform:
Google Vertex AI: Use Vertex AI Pipelines to automate the execution of each step of the ML workflow.
**Key Actions**:
- Containerize the components using Docker for data extraction, cleaning, and model training.
- Develop and test pipeline components using Python (Jupyter notebooks can be helpful here).
- Split your pipeline into separate stages, such as data extraction, cleaning, splitting, and model training​(Assignment 2 Example fr…).
- Store intermediate outputs like training and test datasets in Google Cloud Storage.

Example:
For breast cancer prediction, separate components are used to clean and split the data​(Assignment 1 Example fr…).
In a toxicity classification example, the pipeline consists of data ingestion, cleaning, splitting, training, and prediction components​(Assignment 2 Example fr…).

4. CI/CD Pipelines
Goal: Implement continuous integration and continuous deployment (CI/CD) pipelines to ensure the automation of pipeline execution, model retraining, and deployment.
Platform:
Google Cloud Build: Use Cloud Build for setting up CI/CD pipelines that automate deployment, training, and API exposure.
**Key Actions**:
- Create a YAML-based CI/CD pipeline configuration that defines triggers to re-execute the pipeline when the code changes or new data is uploaded​(Assignment 1 Example fr…).
- Automate retraining and deployment of models using triggers in Cloud Build, which will automatically rebuild the Docker images and redeploy services when changes occur.
Example:
In the previous assignments, CI/CD pipelines triggered model retraining and redeployment based on new data or code changes.

5. Develop the Model as a Service (API)
Goal: Expose your model as a service by creating RESTful APIs.
Platform:
Insomnia: Use Insomnia for API testing.
Google Cloud Run: Deploy the model as a service (using Docker containers) with Google Cloud Run.
**Key Actions**:
- Develop prediction APIs and ensure they can be called via HTTP requests.
- Set up and test endpoints using Insomnia. The endpoints should allow users to make predictions using the model hosted in the cloud​(Assignment 1 Example fr…).
Example:
Previous projects involved creating APIs for serving predictions via Flask apps or custom APIs​.

6. Develop a Prediction UI
Goal: Create a simple front-end interface for users to interact with the model.
Platform:
Flask/HTML/CSS: Build a basic UI using Flask that allows users to input data and view predictions.
**Key Actions**:
- The UI should call the prediction API to get a result and display it to the user​

7. Deploy and Expose the Model
Goal: Deploy the prediction service and make it publicly accessible.
Platform:
Google Cloud Run: Deploy the containerized application, expose it to the public, and ensure the APIs are accessible.
**Key Actions**:
- Ensure your deployment pipeline includes triggers that redeploy the prediction service when there are updates to the model or UI​(Assignment 2 Example fr…).
- Set up public access to the API for external users to make predictions​(Assignment 1 Example fr…).

8. Write the Report
Goal: Summarize your work in a concise report (max 7 pages).
Structure:
Overview of the ML Application:
Describe the goal of your application (e.g., predicting cancer or classifying toxic comments).
Design and Implementation of MLOps:
Detail the pipeline, prediction service, and CI/CD setup​(Assignment 2 Example fr…).
Reflection:
Reflect on challenges, alternative designs, and potential improvements​(Assignment 2 Example fr…).
Individual Contributions:
Detail the roles of each team member in the project.

9. Submission
Goal: Submit the completed assignment.
Platform:
GitHub: Host all source code in a GitHub repository.
Canvas: Submit the report and GitHub link through the Canvas submission system.

