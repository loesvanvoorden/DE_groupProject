import os

from flask import Flask, request

from performance_predictor import PerformancePredictor

import logging
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
app.config["DEBUG"] = True


# Updated app.py from prediction-api-gcs
import os
from flask import Flask, request, render_template, jsonify
from performance_predictor import PerformancePredictor
import logging

app = Flask(__name__)
app.config["DEBUG"] = True

pp = PerformancePredictor()

@app.route('/performance_predictor/model', methods=['PUT'])
def refresh_model():
    try:
        pp.download_model()
        return jsonify({'status': 'Model refreshed successfully'}), 200
    except Exception as e:
        logging.error(f"Model refresh failed: {e}")
        return jsonify({'error': 'Model refresh failed'}), 500

@app.route('/performance_predictor/', methods=['POST'])
def predict_str():
    prediction_input = request.get_json()
    return pp.predict_single_record(prediction_input)

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)



pp = PerformancePredictor()
app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
