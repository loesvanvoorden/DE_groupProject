import os

from flask import Flask, request

from performance_predictor import PerformancePredictor

import logging
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/performance_predictor/model', methods=['PUT'])
def refresh_model():
    try:
        return pp.download_model()
    except Exception as e:
            logging.error(f"Model refresh failed: {e}")
            return jsonify({'error': 'Model refresh failed'}), 500
@app.route('/performance_predictor/', methods=['POST'])  # path of the endpoint. Except only HTTP POST request
def predict_str():
    # the prediction input data in the message body as a JSON payload
    prediction_inout = request.get_json()
    return pp.predict_single_record(prediction_inout)

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)



pp = PerformancePredictor()
app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
