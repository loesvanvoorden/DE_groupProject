import os

from flask import Flask, request

from peformance_predictor import PerformancePredictor

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/performance_predictor/model', methods=['PUT'])  # trigger updating the model
def refresh_model():
    return pp.download_model()


@app.route('/performance_predictor/', methods=['POST'])  # path of the endpoint. Except only HTTP POST request
def predict_str():
    # the prediction input data in the message body as a JSON payload
    prediction_inout = request.get_json()
    return pp.predict_single_record(prediction_inout)


pp = PerformancePredictor()
app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
