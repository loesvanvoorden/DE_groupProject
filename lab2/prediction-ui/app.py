# importing Flask and other modules
import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

# Flask constructor
app = Flask(__name__)

# Set up logging configuration for debugging
logging.basicConfig(level=logging.DEBUG)

# A decorator used to tell the application which URL is associated function
@app.route('/checkperformance', methods=["GET", "POST"])
def check_performance():
    if request.method == "GET":
        return render_template("input_form_page.html")
    
    elif request.method == "POST":
        try:
            # Capture input from the form
            prediction_input = [
                {
                    "schoolsup": str(request.form.get("schoolsup")),   
                    "higher": str(request.form.get("higher")),  
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

            # Ensure that the predictor API URL is set in the environment variables
            predictor_api_url = os.environ.get('PREDICTOR_API')
            if not predictor_api_url:
                logging.error("PREDICTOR_API environment variable is not set.")
                return jsonify(message="PREDICTOR_API not configured"), 500
            
            # Log the API URL for debugging purposes
            logging.debug("Using predictor API URL: %s", predictor_api_url)

            # Make the API request
            res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))
            
            # Log the raw response content for debugging
            logging.debug("Raw API response: %s", res.text)

            # Check for non-200 status codes
            if res.status_code != 200:
                logging.error("Non-200 status code from API: %d", res.status_code)
                return jsonify(message=f"Error: Received status code {res.status_code}"), 500

            # Attempt to decode the JSON response
            try:
                prediction_value = res.json().get('result', None)
                if prediction_value:
                    logging.info("Prediction Output : %s", prediction_value)
                    return render_template("response_page.html", prediction_variable=prediction_value)
                else:
                    logging.error("No 'result' key found in the API response")
                    return jsonify(message="Error: No result in the API response"), 500
            except json.JSONDecodeError:
                logging.error("Failed to decode JSON from API response")
                return jsonify(message="Error: Invalid JSON in API response"), 500
        
        except Exception as e:
            logging.exception("Error during performance check")
            return jsonify(message="Error: An exception occurred"), 500

    else:
        return jsonify(message="Method Not Allowed"), 405


# The code within this conditional block will only run if this file is executed directly
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
