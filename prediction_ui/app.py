from flask import Flask, render_template

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config["DEBUG"] = True

@app.route('/input')
def ml_application():
    return render_template("input.html")

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')