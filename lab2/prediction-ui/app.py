import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route('/checkperformance', methods=["GET", "POST"])
def check_performance():
    if request.method == "GET":
        return render_template("input_form_page.html")

    elif request.method == "POST":
        prediction_input = [
            {
                "schoolsup": int(request.form.get("schoolsup")),
                "higher": int(request.form.get("higher")),
                "absences": int(request.form.get("absences")),
                "failures": int(request.form.get("failures")),
                "Medu": int(request.form.get("Medu")),
                "Fedu": float(request.form.get("Fedu")),
                "Walc": float(request.form.get("Walc")),
                "Dalc": int(request.form.get("Dalc")),
                "famrel": int(request.form.get("famrel")),
                "goout": int(request.form.get("goout")),
                "freetime": int(request.form.get("freetime")),
                "studytime": int(request.form.get("studytime"))
            }
        ]

        logging.debug("Prediction input : %s", prediction_input)

        predictor_api_url = os.environ['PREDICTOR_API']
        try:
            res = requests.post(predictor_api_url, json=prediction_input)
            res.raise_for_status()
            prediction_value = res.text
            logging.info("Prediction Output : %s", prediction_value)
            return render_template("response_page.html", prediction_variable=prediction_value)
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return jsonify({'error': 'API request failed'}), 500

    else:
        return jsonify(message="Method Not Allowed"), 405

# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)