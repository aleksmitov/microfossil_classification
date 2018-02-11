import functools
import tensorflow as tf
import numpy as np
import functools
import json
import cv2
import time
import os
import csv

import neural_networks.alex_net

IMAGE_SOURCE_DIR = "./unprocessed_crops"
IMAGE_DEST_DIR = "./processed_crops"
METADATA_FILE = "model_metadata.json"
CSV_OUTPUT = "records.csv"
MODEL_TO_LOAD = "Alex_Net_classes_2_channels_3_batch_512_learning_rate_1e-05/step_3300_Alex_Net_classes_2_channels_3_batch_512_learning_rate_0.0001"
HIGH_CONFIDENCE = 0.85


def define_computational_graph(input_dims, net_architecture, number_of_classes):
    """
    Performs prediction on the inputs for the given model
    :param inputs: Numpy matrix with shape [images_count, height, width, channels]
    :param net_architecture: a function defining the network architecture
    :param number_of_classes: an int
    :return: a numpy matrix with softmax-ed output scores
    """

    print("Defining the net architecture...")

    with tf.variable_scope("inputs"):
        x = tf.placeholder(dtype=tf.uint8, shape=input_dims, name="x")
        x_transformed = tf.cast(x, tf.float32) / 255  # Convert to [0, 1] range
        dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_rate")

    with tf.variable_scope("net_architecture"):
        # Getting net output scores and name
        y_prime, network_name = net_architecture(x_transformed, number_of_classes, dropout_rate, is_training=False)
        y_prime = tf.nn.softmax(y_prime)  # Squish scores in [0, 1]

    # saver for checkpoints
    model_variables = tf.global_variables("net_architecture")
    saver = tf.train.Saver(model_variables)

    return x, y_prime, dropout_rate, saver


def predict(inputs, x, y_prime, dropout, session):
    """
    Passes the input through the computational graph
    :param inputs: 4D numpy array with shape [image_count, height, width, channels]
    :param x: tf.placeholder for the input
    :param y_prime: tf.placeholder for the network output
    :param dropout: tf.placeholder for the dropout probability
    :param session: tf.Session instance
    :return: numpy array with the confidence scores
    """
    network_output_scores = session.run(y_prime, feed_dict={x: inputs, dropout: 0})
    return network_output_scores


def read_metadata(metadata_file):
    """
    Parses the metadata JSON file
    :param metadata_file: string with the file path
    :return: a tuple with metadata for the training and testing set
    """
    with open(metadata_file, 'r') as fp:
        metadata = json.load(fp)
        input_image_dims = tuple(metadata["input_image_dims"])
        models_dir = metadata["models_dir"]
        training_set = metadata["training_set"]
        number_of_classes = len(training_set)

    return models_dir, number_of_classes, input_image_dims


def classify_microfossils(source_dir, destination_dir, prediction_func, input_image_dims, confidence_threshold,
                                                read_grayscale_images=False, recursive_destination_structure=True):
    """
    Recursively traverses the source dir and classifies every image with the prediction function
    :param source_dir: directory file path
    :param destination_dir: destination file path
    :param prediction_func: a function taking a 4D numpy array as input
    :param input_image_dims: a tuple of (height, width) of the images
    :param confidence_threshold: a number in the range [0, 1]
    :param read_grayscale_images: boolean value
    :param recursive_destination_structure: boolean, if True destination dir will have the same structure as source
    :return: a list of tuples (image path, confidence scores) for every processed image
    """
    if os.path.isdir(source_dir) is False:
        raise Exception("Not a valid source path")
    if os.path.isdir(destination_dir) is False:
        os.makedirs(destination_dir)

    print("Currently processing images in dir: {}".format(source_dir))
    image_extensions = [".tif", ".png", ".jpg", "jpeg"]
    sub_dirs = []
    images_in_dir = []
    for file in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, file)) and os.path.join(source_dir, file) != destination_dir:
            sub_dirs.append(file)
        # If it's an image with the given extensions
        elif reduce((lambda x, y: x or y), [file.lower().endswith(ext) for ext in image_extensions]):
            images_in_dir.append(file)

    records = []  # Keep a list of processing results

    # Now process the images
    for image_path in images_in_dir:
        full_image_path = os.path.join(source_dir, image_path)
        cv2_format_flag = cv2.IMREAD_GRAYSCALE if read_grayscale_images else cv2.IMREAD_COLOR
        image = cv2.imread(full_image_path, cv2_format_flag)
        if image is None:
            print("Couldn't read image and was skipped: {}".format(full_image_path))
            continue
        if image.shape[0] != input_image_dims[0] or image.shape[1] != input_image_dims[1]:
            image = cv2.resize(image, (input_image_dims[1], input_image_dims[0]))
        if read_grayscale_images:
            image = image.reshape(image.shape + (1,))  # Add additional channel

        input_vector = image.reshape((1,) + image.shape)
        predictions = prediction_func(input_vector)

        # Output logic here
        trimmed_path = os.path.join(source_dir, image_path).replace("./", "")
        records.append((trimmed_path, predictions[0]))
        classified_image_path = trimmed_path.replace("/", ".")

        target_class = -1
        for class_i in range(0, predictions.shape[-1]):
            confidence_score = predictions[0, class_i]
            if confidence_score > confidence_threshold:
                target_class = class_i
                break

        classification_dir = "Class_{}".format(str(target_class)) if target_class != -1 else "Mixed"
        if os.path.isdir(os.path.join(destination_dir, classification_dir)) is False:
            os.makedirs(os.path.join(destination_dir, classification_dir))

        cv2.imwrite(os.path.join(destination_dir, classification_dir, classified_image_path), image)

    # Recursively apply to all subdirs
    for subdir in sub_dirs:
        source_subdir = os.path.join(source_dir, subdir)
        destination_subdir = os.path.join(destination_dir, subdir)
        target_destination = destination_subdir if recursive_destination_structure else destination_dir
        sub_records = classify_microfossils(source_subdir, target_destination, prediction_func, input_image_dims,
                                        confidence_threshold, read_grayscale_images, recursive_destination_structure)
        records.extend(sub_records)

    return records


def main():
    models_dir, number_of_classes, input_image_dims = read_metadata(METADATA_FILE)
    neural_net = neural_networks.alex_net.alex_net
    model_path = os.path.join(models_dir, MODEL_TO_LOAD)
    input_dims = (1,) + input_image_dims + (3,)  # Dimensions of the input tensor for the neural net

    x, y_prime, dropout, saver = define_computational_graph(input_dims=input_dims, net_architecture=neural_net,
                                                                                number_of_classes=number_of_classes)
    print("Starting prediction...")
    with tf.Session() as session, open(CSV_OUTPUT, "wb") as csv_file:
        prediction_func = functools.partial(predict, x=x, y_prime=y_prime, dropout=dropout, session=session)
        session.run(tf.global_variables_initializer())
        saver.restore(session, model_path)
        print("Loaded model weights: {}".format(model_path))
        records = classify_microfossils(IMAGE_SOURCE_DIR, IMAGE_DEST_DIR, prediction_func, input_image_dims,
                                HIGH_CONFIDENCE, read_grayscale_images=False, recursive_destination_structure=False)
        print("Writing csv output...")
        writer = csv.writer(csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        for record_image_path, record_image_classification in records:
            writer.writerow([record_image_path, record_image_classification.round(decimals=2)])
    print("DONE")


main()


