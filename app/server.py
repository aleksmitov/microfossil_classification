from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from threading import Thread
from multiprocessing import Process
import os
import cv2
import shutil
import json
import csv
import time
import numpy as np
import tensorflow as tf
import functools
import tempfile
import uuid
import zipfile
from PIL import Image

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extract_microfossils
import neural_networks
import predict
from Metadata import Metadata

ALLOWED_INPUT_IMAGE_EXTENSIONS = set(["png", "tif", "jpg", "jpeg", "bmp"])
ALLOWED_INPUT_ARCHIVE_EXTENSIONS = set(["zip"])

MIN_MICROFOSSIL_SIZE = 2
METADATA_FILE = "../model_metadata.json"
MODEL_TO_LOAD = "Alex_Net_classes_2_channels_3_batch_512_learning_rate_1e-05/Alex_Net_classes_2_channels_3"
CROP_DIMS = (200, 200)
ARCHIVES_DIR = "./uploads/archives"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./static/uploads"
app.secret_key = "secret_key"


@app.route("/")
def index():
    return "Index!"


@app.route("/batch_extract_and_classify", methods=["GET", "POST"])
def batch_extract_and_classify():
    start_time = time.time()
    uploaded_archive_link = None

    if request.method == "POST":
        # check if the post request has the file part
        if "archive" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["archive"]
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)
        if file is None or is_file_allowed(file.filename, ALLOWED_INPUT_ARCHIVE_EXTENSIONS) is False or \
                                                                        zipfile.is_zipfile(file.stream) is False:
            flash("Invalid file or extension!")
            return redirect(request.url)

        zip_object = zipfile.ZipFile(file.stream)
        batch_process = BatchProcessingProcess(zip_object)
        batch_process.start()
        batch_process.join(0.0001)
        flash("Archive uploaded successfully for processing!")

    elapsed_time = time.time() - start_time

    return render_template("batch_image_extraction_and_classification.html",
                                    elapsed_time=elapsed_time)



@app.route("/extract_and_classify_single_image", methods=["GET", "POST"])
def extract_and_classify():
    start_time = time.time()
    unfiltered_images_paths_predictions = []
    filtered_images_paths_predictions = []
    uploaded_image_link = None

    if request.method == "POST":
        # check if the post request has the file part
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["image"]
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)
        if file is None or is_file_allowed(file.filename, ALLOWED_INPUT_IMAGE_EXTENSIONS) is False:
            flash("Invalid file or extension!")
            return redirect(request.url)
        extension = file_extension(file.filename)
        unique_filename = str(uuid.uuid4()) + ".png"
        image_object = Image.open(file.stream)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        # Also converts to specified image format
        image_object.save(file_path)
        uploaded_image_link = url_for("static", filename="uploads/{}".format(unique_filename), _external=True)
        #file.save(file_path)

        # Now extract microfossils and classify
        metadata = Metadata(METADATA_FILE)
        model_path = os.path.join("..", (os.path.join(metadata.models_dir, MODEL_TO_LOAD)))
        unfiltered_predictions, filtered_predictions = predict.extract_and_classify_image(file_path, metadata,
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

        flash("Successfully processed image!")

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


class BatchProcessingThread(Thread):
    def __init__(self, val):
        ''' Constructor. '''

        Thread.__init__(self)
        self.val = val

    def run(self):
        while True:
            print("Running thread")
        extension = file_extension(file.filename)
        unique_dirname = str(uuid.uuid4())
        os.makedirs(unique_dirname)  # Create the dir


class BatchProcessingProcess(Process):
    def __init__(self, zip_object, archives_dir, crops_dims, min_microfossil_size, confidence_threshold,
                 model_metadata, model_weights):
        Process.__init__(self)
        self.zip_object = zip_object
        self.archives_dir = archives_dir
        self.crop_dims = crops_dims
        self.min_microfossil_size = min_microfossil_size
        self.confidence_threshold = confidence_threshold
        self.model_metadata = model_metadata
        self.model_weights = model_weights
        self.clean_particles = True

    def run(self):
        unique_dirname = str(uuid.uuid4())
        working_dir_path = os.path.join(self.archives_dir, unique_dirname)
        os.makedirs(working_dir_path)  # Create the dir

        unique_dirname = str(uuid.uuid4())
        extraction_dir_path = os.path.join(self.archives_dir, unique_dirname)
        os.makedirs(working_dir_path)  # Create the dir

        self.zip_object.extractall(working_dir_path)
        extract_microfossils.extract_microfossils_in_dir(working_dir_path, extraction_dir_path, self.crop_dims,
                                                                self.min_microfossil_size, self.clean_particles)
        #shutil.rmtree(working_dir_path)
        print("Working dir to delete: {}".format(working_dir_path))
        # Dir ready for processing

        neural_net = neural_networks.alex_net.alex_net
        model_path = os.path.join(self.model_metadata.models_dir, self.model_weights)
        input_image_dims = self.model_metadata.input_image_dims
        input_dims = (1,) + input_image_dims + (3,)  # Dimensions of the input tensor for the neural net

        x, y_prime, dropout, saver = predict.define_computational_graph(input_dims=input_dims,
                                                        net_architecture=neural_net,
                                                        number_of_classes=self.model_metadata.number_of_classes)
        records_file_path = os.path.join(extraction_dir_path, "records.csv")

        print("Starting classification...")
        with tf.Session() as session, open(records_file_path, "wb") as csv_file:
            prediction_func = functools.partial(predict, x=x, y_prime=y_prime, dropout=dropout, session=session)
            session.run(tf.global_variables_initializer())
            saver.restore(session, model_path)
            print("Loaded model weights: {}".format(model_path))
            records = predict.classify_microfossils(working_dir_path, extraction_dir_path, prediction_func,
                                                    self.crop_dims,
                                                    self.confidence_threshold, False,
                                                    recursive_destination_structure=True)
            print("Writing csv output...")
            writer = csv.writer(csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            for record_image_path, record_image_classification in records:
                writer.writerow([record_image_path, record_image_classification.round(decimals=2)])
        print("Done with classification output")

        # Now zip extraction dir
        








