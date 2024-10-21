import requests
import json

# Load the training data from the JSON file
with open('data/train_data.json', 'r') as f:
    train_data = json.load(f)

# Send POST request to the Flask API
response = requests.post('http://localhost:5000/training-api/model', json=train_data)

# Print the response from the server
print(response.json())