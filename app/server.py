from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
import os
import cv2
import json
import time
import numpy as np
import tensorflow as tf
import functools
import tempfile
import uuid
from PIL import Image

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extract_microfossils
import neural_networks
import predict
from Metadata import Metadata

ALLOWED_EXTENSIONS = set(["png", "tif"])

MIN_MICROFOSSIL_SIZE = 2
METADATA_FILE = "../model_metadata.json"
MODEL_TO_LOAD = "Alex_Net_classes_2_channels_3_batch_512_learning_rate_1e-05/Alex_Net_classes_2_channels_3"
CROP_DIMS = (200, 200)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./static/uploads"
app.secret_key = "secret_key"

@app.route("/")
def index():
    return "Index!"


@app.route("/extract_and_classify_single_image", methods=["GET", "POST"])
def extract_and_classify():
    start_time = time.time()
    unfiltered_images_paths_predictions = []
    filtered_images_paths_predictions = []
    uploaded_image_link = None

    if request.method == "POST":
        # check if the post request has the file part
        if "image" not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files["image"]
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and is_file_allowed(file.filename, ALLOWED_EXTENSIONS):
            extension = file_extension(file.filename)
            unique_filename = str(uuid.uuid4()) + ".png"
            image_object = Image.open(file.stream)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            # Also converts to desired image format
            image_object.save(file_path)
            #file.save(file_path)
        metadata = Metadata(METADATA_FILE)
        model_path = os.path.join("..", (os.path.join(metadata.models_dir, MODEL_TO_LOAD)))
        image_file = "./test.png"
        unfiltered_predictions, filtered_predictions = predict.extract_and_classify_image(image_file, metadata,
                                                                        model_path, MIN_MICROFOSSIL_SIZE, CROP_DIMS, True)

        for crop, crop_predictions in unfiltered_predictions:
            unique_filename = str(uuid.uuid4()) + ".png"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            cv2.imwrite(file_path, crop)
            link_to_file = url_for("static", filename="uploads/{}".format(unique_filename), _external=True)
            unfiltered_images_paths_predictions.append((link_to_file, crop_predictions.tolist()))

        for crop, crop_predictions in filtered_predictions:
            unique_filename = str(uuid.uuid4()) + ".png"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            cv2.imwrite(file_path, crop)
            link_to_file = url_for("static", filename="uploads/{}".format(unique_filename), _external=True)
            filtered_images_paths_predictions.append((link_to_file, crop_predictions.tolist()))

    elapsed_time = time.time() - start_time
    return render_template("image_extraction_and_classification.html",
                                    unfiltered_predictions=unfiltered_images_paths_predictions,
                                    filtered_predictions=filtered_images_paths_predictions,
                                    uploaded_image_link=uploaded_image_link,
                                    elapsed_time=elapsed_time)


def is_file_allowed(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def file_extension(filename):
    return filename.rsplit('.', 1)[1].lower()





