from flask import Flask, render_template

app = Flask(__name__)

@app.route("/prediction")
def prediction():
    return render_template('prediciton.html')