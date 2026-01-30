from flask import Flask, render_template, request
import joblib

# Create Flask app
app = Flask(__name__)

# Load ML model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        message = request.form["message"]

        message_vector = vectorizer.transform([message])
        result = model.predict(message_vector)

        prediction = result[0]

    return render_template("index.html", prediction=prediction)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

