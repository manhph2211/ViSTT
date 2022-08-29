import random
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    audio_file = request.files["file"]
    predicted_keyword = process(audio_file)
    data = {"keyword": predicted_keyword}
    return jsonify(data)


def process(audio_path):
    return ""


if __name__ == "__main__":
    app.run(debug=False)













