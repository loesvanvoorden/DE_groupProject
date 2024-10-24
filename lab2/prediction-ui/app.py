# importing Flask and other modules
import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

# Flask constructor
app = Flask(__name__)


# A decorator used to tell the application
# which URL is associated function
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
        logging.info("Prediction Input: %s", prediction_input)

        # -----------------------------------------

        logging.debug("Prediction input : %s", prediction_input)

        # use requests library to execute the prediction service API by sending an HTTP POST request
        # use an environment variable to find the value of the diabetes prediction API
        # json.dumps() function will convert a subset of Python objects into a json string.
        # json.loads() method can be used to parse a valid JSON string and convert it into a Python Dictionary.
        predictor_api_url = os.environ['PREDICTOR_API']

        # This is just to test
        res = requests.post(predictor_api_url, json=prediction_input)
        try:
            prediction_value = res.json().get('result')
        except requests.exceptions.JSONDecodeError:
            logging.error("Response is not valid JSON: %s", res.text)
            return jsonify(message="Prediction API returned an invalid response"), 500


        res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))

        prediction_value = res.json()['result']
        logging.info("Prediction Output : %s", prediction_value)
        return render_template("response_page.html",
                               prediction_variable=prediction_value)

    else:
        return jsonify(message="Method Not Allowed"), 405  # The 405 Method Not Allowed should be used to indicate
    # that our app that does not allow the users to perform any other HTTP method (e.g., PUT and  DELETE) for
    # '/checkdiabetes' path


# The code within this conditional block will only run the python file is executed as a
# script. See https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
