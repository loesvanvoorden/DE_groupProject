import os
import logging
from flask import Flask, request, render_template, jsonify
from performance_predictor import PerformancePredictor

# Initialize Flask app
app = Flask(__name__)
app.config["DEBUG"] = True

# Create an instance of PerformancePredictor before defining routes
pp = PerformancePredictor()

# Route to refresh the model
@app.route('/performance_predictor/model', methods=['PUT'])
def refresh_model():
    try:
        return pp.download_model()
    except Exception as e:
        logging.error(f"Model refresh failed: {e}")
        return jsonify({'error': 'Model refresh failed'}), 500

# Route to make a prediction
@app.route('/performance_predictor/', methods=['POST'])
def predict_str():
    prediction_input = request.get_json()
    return pp.predict_single_record(prediction_input)

# Run the app
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)