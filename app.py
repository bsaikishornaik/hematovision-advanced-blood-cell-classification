import os
import numpy as np
import cv2
import base64
from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

model = load_model("Blood Cell.h5")

class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']


def predict_image_class(image_path, model):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)[0]
    idx = np.argmax(predictions)
    confidence = float(predictions[idx]) * 100

    return class_labels[idx], round(confidence, 2), img_rgb


@app.route("/")
@app.route("/home.html")
def home():
    return render_template("Home.html")


@app.route("/project.html")
def project():
    return render_template("Project.html")


@app.route("/team.html")
def team():
    return render_template("team.html")


@app.route("/result.html", methods=["GET", "POST"])
def result():

    if request.method == "POST":

        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        upload_dir = os.path.join("static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        label, confidence, img_rgb = predict_image_class(file_path, model)

        _, img_encoded = cv2.imencode(
            ".png",
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        )
        img_str = base64.b64encode(img_encoded).decode("utf-8")

        return render_template(
            "result.html",
            class_label=label,
            confidence=confidence,
            img_data=img_str
        )

    return render_template("result.html")


if __name__ == "__main__":
    app.run(debug=True)

