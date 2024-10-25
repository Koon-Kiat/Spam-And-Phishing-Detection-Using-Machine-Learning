from flask import Flask
from flask import render_template
from flask.helpers import send_file
from flask import request

import single_model_evaluation

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/taskpane.html")
def taskpane():
    return render_template("taskpane.html")


@app.route("/evaluateEmail", methods=['POST'])
def evaluateEmail():
    # request.get_data() gets the request body (it will be the email in EML)
    output = single_model_evaluation.main(request.get_data())
    return output


@app.route("/commands.html")
def commands():
    return render_template("commands.html")


@app.route("/assets/icon-16.png")
def icon16():
    return send_file("./static/assets/icon-16.png", mimetype='image/png')


@app.route("/assets/icon-32.png")
def icon32():
    return send_file("./static/assets/icon-32.png", mimetype='image/png')


@app.route("/assets/icon-64.png")
def icon64():
    return send_file("./static/assets/icon-64.png", mimetype='image/png')


@app.route("/assets/icon-80.png")
def icon80():
    return send_file("./static/assets/icon-80.png", mimetype='image/png')


@app.route("/assets/icon-128.png")
def icon128():
    return send_file("./static/assets/icon-128.png", mimetype='image/png')


@app.route("/assets/logo-filled.png")
def iconlogofilled():
    return send_file("./static/assets/logo-filled.png", mimetype='image/png')


@app.route('/favicon.ico')
def favicon():
    return send_file('./static/favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == "__main__":
    app.run(debug=True)
