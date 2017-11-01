import tensorflow as tf
import numpy as np
import functools
import math
import json
import cv2
import time
import os

from stochastic_data_retriever import StochasticDataRetriever
import neural_networks.test_net
import neural_networks.alex_net

METADATA_FILE = "model_metadata.json"
BRIGHTNESS_DELTA = 0.25  # Brightness delta range during augmentation
ROTATION_ANGLE_RANGE = (-15, 15)  # Up to how much degrees to rotate from ideal square fit during augmentation
LEARNING_RATE = 0.00001  # Used during training for updating the model weights
BATCH_SIZE = 512
DROPOUT_RATE = 0.80
TRAINING_EPOCHS = 1000000
CHECKPOINTS_TO_KEEP = 5  # Keep the last N weight checkpoints during training
LOGS_DIR = "training_logs"
TRAIN_FROM_SAVED_MODEL = True
MODEL_TO_LOAD = "step_3300_Alex_Net_classes_2_channels_3_batch_512_learning_rate_0.0001"  # Used if train_from_saved_model is True
LOADED_MODEL_TRAIN_FROM_STEP = 3301  # If training from a saved model, start from this training step
LOG_FREQUENCY = 50
VALIDATION_FREQUENCY = 50
SAVE_MODEL_FREQUENCY = 100


def main():
    #test_augmentation()
    print("Reading metadata...")
    training_set, testing_set, input_image_dims, models_dir = read_metadata(METADATA_FILE)
    print("Initialising training data retriever...")
    training_data_retriever = StochasticDataRetriever(training_set, input_image_dims, grayscale=False,
                                                      sample_evenly_across_classes=True, resize_images=True,
                                                      load_in_memory=True)
    print("Initialising testing data retriever...")
    testing_data_retriever = StochasticDataRetriever(testing_set, input_image_dims, grayscale=False,
                                                     sample_evenly_across_classes=True, resize_images=True,
                                                     load_in_memory=True)

    #neural_net = neural_networks.test_net.test_net  # Sample net architecture
    neural_net = neural_networks.alex_net.alex_net  # AlexNet architecture
    augmentation_func = functools.partial(augmentation_graph, brightness_delta=BRIGHTNESS_DELTA,
                                                                    rotation_range=ROTATION_ANGLE_RANGE)
    print("Calling the training function...")
    train(neural_net, training_data_retriever, testing_data_retriever, BATCH_SIZE, TRAINING_EPOCHS, augmentation_func,
                LEARNING_RATE, DROPOUT_RATE, CHECKPOINTS_TO_KEEP, LOGS_DIR, models_dir, LOG_FREQUENCY,SAVE_MODEL_FREQUENCY,
                VALIDATION_FREQUENCY, TRAIN_FROM_SAVED_MODEL, MODEL_TO_LOAD, LOADED_MODEL_TRAIN_FROM_STEP)

    print("Done training.")


def train(net_architecture, training_data_retriever, testing_data_retriever, batch_size, training_epochs,
            augmentation_graph, learning_rate, dropout, checkpoints_to_keep, logs_dir, models_dir,
            log_frequency, save_model_frequency,validation_frequency,
            train_from_saved_model=False, model_to_load=None, loaded_model_train_from_step=0):
    data_channels = training_data_retriever.get_image_channels()
    input_shape = (batch_size,) + training_data_retriever.get_data_shape()
    number_of_classes = training_data_retriever.get_number_of_classes()

    print("Defining the net architecture...")

    with tf.variable_scope("inputs"):
        x = tf.placeholder(dtype=tf.uint8, shape=input_shape, name="x")
        y = tf.placeholder(dtype=tf.int16, shape=(batch_size, number_of_classes), name="y")
        y = tf.stop_gradient(y)  # Stop backpropagating through y
        dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_rate")

    with tf.variable_scope("augmentation"):
        augmented_sample = augmentation_graph(x, input_shape)  # Build the augmentation graph
        augmented_sample = tf.cast(augmented_sample, tf.float32) / 255  # Convert to [0, 1] range

    with tf.variable_scope("net_architecture"):
        # Getting net output scores and name
        y_prime, network_name = net_architecture(augmented_sample, number_of_classes, dropout_rate, True)

    with tf.variable_scope("metrics"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                logits=y_prime, name="cross_entropy"), name="mean_entropy")
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_prime, 1), name="prediction")
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        y_labels = tf.argmax(y, axis=1, name="y_labels")
        y_prime_softmax = tf.nn.softmax(y_prime)  # Squish activation scores between [0,1]
        predictions = []
        for class_i in range(0, number_of_classes):
            i_class_prediction_indexes = tf.equal(y_labels, class_i)
            i_class_predictions = tf.boolean_mask(y_prime_softmax, i_class_prediction_indexes,
                                                                name="true_class_{}_predictions".format(class_i))
            for class_j in range(0, number_of_classes):
                # If class i is the true class, take the predictions for class j
                true_i_class_j_predictions = i_class_predictions[:, class_j]
                predictions.append((class_i, class_j, true_i_class_j_predictions))

    with tf.variable_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, name="adam_optimizer")

    with tf.variable_scope("summaries"):
        loss_summary = tf.summary.scalar("Loss", cross_entropy)
        acc_summary = tf.summary.scalar("Accuracy", accuracy)
        class_prediction_summaries = []
        for true_class_i, predicted_class_j, i_j_predictions in predictions:
            i_j_prediction_summary = tf.summary.histogram("true_class_{}_predictions_for_class_{}".format(
                                                                true_class_i, predicted_class_j), i_j_predictions)
            class_prediction_summaries.append(i_j_prediction_summary)

        input_images_summary = tf.summary.image("Augmented_images", augmented_sample, max_outputs=10)
        validation_summary = tf.summary.merge([input_images_summary, loss_summary,
                                                    acc_summary] + class_prediction_summaries)
        training_summary = tf.summary.merge([input_images_summary, loss_summary,
                                                    acc_summary] + class_prediction_summaries)
        #test_summary = tf.summary.merge([input_images_summary, acc_summary])

    # Define working directories
    logs_dir = os.path.join(logs_dir, "{}_classes_{}_channels_{}_batch_{}_learning_rate_{}".format(
        network_name, number_of_classes, data_channels, batch_size, learning_rate))
    models_dir = os.path.join(models_dir, "{}_classes_{}_channels_{}_batch_{}_learning_rate_{}".format(
        network_name, number_of_classes, data_channels, batch_size, learning_rate))
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # saver for checkpoints
    model_variables = tf.global_variables("net_architecture")
    saver = tf.train.Saver(model_variables, max_to_keep=checkpoints_to_keep)

    print("Starting training on {}...".format(network_name))

    with tf.Session() as session:
        summary_writer = tf.summary.FileWriter(logs_dir + "_train", session.graph)
        summary_writer_validation = tf.summary.FileWriter(logs_dir + "_validate", session.graph)

        session.run(tf.global_variables_initializer())
        starting_step = 0

        if train_from_saved_model:
            model_path = os.path.join(models_dir, model_to_load)
            saver.restore(session, model_path)
            starting_step = loaded_model_train_from_step
            print("Loaded model weights: {}".format(model_path))

        tf.get_default_graph().finalize()  # Graph is complete, a safety measure

        # Training and validation
        start_time = time.time()
        for training_step in range(starting_step, training_epochs):
            sample_x, sample_y = training_data_retriever.sample_data(batch_size)  # Sample from training set

            _, summary_str, loss, accuracy_val = session.run([train_step, training_summary, cross_entropy, accuracy],
                                                    feed_dict={x: sample_x, y: sample_y, dropout_rate: dropout})

            #test_augmentation(sample_x, a, in_floating_point=True)  # Some testing

            if training_step % log_frequency == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                summary_writer.add_summary(summary_str, training_step)

                print("Training step {}, loss: {}, accuracy: {}".format(training_step, loss, accuracy_val))
                print("Time per step: {:.2f}sec.".format(elapsed_time*1.0/log_frequency))

            # Monitoring accuracy using test set
            if training_step % validation_frequency == 0:
                test_sample_x, test_sample_y = testing_data_retriever.sample_data(batch_size)
                validation_accuracy, summary_str = session.run([accuracy, validation_summary],
                                                    feed_dict={x: test_sample_x, y: test_sample_y, dropout_rate: 0})

                print("Training step {}, accuracy on validation batch: {}".format(training_step, validation_accuracy))
                summary_writer_validation.add_summary(summary_str, training_step)

            # Save the model checkpoint periodically.
            if training_step % save_model_frequency == 0:
                model_name = "step_{}_{}_classes_{}_channels_{}_batch_{}_learning_rate_{}".format(training_step,
                                            network_name, number_of_classes, data_channels, batch_size, learning_rate)
                checkpoint_path = os.path.join(models_dir, model_name)
                saver.save(session, checkpoint_path)
                print("Training step {}, saved model in: {}".format(training_step, checkpoint_path))


def read_metadata(metadata_file):
    """
    Parses the metadata JSON file
    :param metadata_file: string with the file path
    :return: a tuple with metadata for the training and testing set
    """
    with open(metadata_file, 'r') as fp:
        metadata = json.load(fp)
        input_image_dims = metadata["input_image_dims"]
        training_set = metadata["training_set"]
        testing_set = metadata["testing_set"]
        models_dir = metadata["models_dir"]
        training_set_tuples = []
        testing_set_tuples = []
        for data_row in training_set:
            training_set_tuples.append(tuple(data_row))
        for data_row in testing_set:
            testing_set_tuples.append(tuple(data_row))

    return training_set_tuples, testing_set_tuples, input_image_dims, models_dir


def augmentation_graph(input_tensor, data_shape, brightness_delta, rotation_range):
    """
    Constructs the graph for augmenting the input data
    :param input_tensor: tf.Tensor with input data
    :param data_shape: a tuple with the data dimensions
    :param brightness_delta: a number with the bound on brightness difference
    :param rotation_range: (lower_bound, higher_bound) tuple for the rotation angle range
    :return: tf.Tensor-s for the input and output of the augmentation graph
    """
    num_samples = data_shape[0]
    augmented_tensor = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_tensor, name="random_flip")
    augmented_tensor = tf.map_fn(lambda img: tf.image.random_brightness(img, brightness_delta),
                                                                        augmented_tensor, name="random_brightness")
    rotation_range = rotation_range[0] * math.pi / 180, rotation_range[1] * math.pi / 180  # Convert to radians
    random_angles = tf.random_uniform([num_samples], rotation_range[0], rotation_range[1], tf.float32)
    right_angle_multiplier = tf.random_uniform([num_samples], 0, 4, tf.int32)  # 90 degree multiplier

    random_angles += tf.multiply(tf.to_float(right_angle_multiplier), math.pi/2)  # Add 0/90/180/270 degrees to angle
    augmented_tensor = tf.contrib.image.rotate(augmented_tensor, random_angles, interpolation='BILINEAR')
    individual_brightness_delta = tf.random_uniform(data_shape, -2, 3, tf.int32)  # For each pixel
    augmented_tensor = tf.clip_by_value(tf.cast(augmented_tensor, tf.int32)+individual_brightness_delta,
                                                                        clip_value_min=0, clip_value_max=255)

    return augmented_tensor


def test_data_receiver():
    training_set, testing_set = read_metadata(METADATA_FILE)
    training_data_retriever = StochasticDataRetriever(training_set, (200, 200), grayscale=True,
                                            sample_evenly_across_classes=True, resize_images=True, load_in_memory=True)
    print ("Data size: ", training_data_retriever.data_size)
    sample_x, sample_y = training_data_retriever.sample_data(1000)
    sum = np.sum(sample_y, axis=0)
    print("Sum: ", sum, sum.shape)
    print (sample_x.shape)
    print (sample_y.shape)


def test_augmentation(sample_x=None, augmented_sample=None, in_floating_point=True):
    if sample_x is None or augmented_sample is None:
        training_set, testing_set, _, _ = read_metadata(METADATA_FILE)
        training_data_retriever = StochasticDataRetriever(training_set, (200, 200), grayscale=True,
                                                          sample_evenly_across_classes=True, resize_images=True,
                                                          load_in_memory=True)
        sample_x, sample_y = training_data_retriever.sample_data(1000)
        start_time = time.time()
        augmentation_input = tf.placeholder(tf.uint8, shape=sample_x.shape)
        augmented_tensor = augmentation_graph(augmentation_input,sample_x.shape, BRIGHTNESS_DELTA, ROTATION_ANGLE_RANGE)
        with tf.Session() as session:
            augmented_sample = session.run(augmented_tensor, feed_dict={augmentation_input: sample_x})
        elapsed = time.time() - start_time
        augmented_sample = augmented_sample.astype(np.uint8)
        print("Augmentation took {} sec.".format(elapsed))
        print("SHAPE: ", augmented_sample.shape, "type: ", augmented_sample.dtype)
    elif in_floating_point is True:
        #sample_x = (sample_x * 255).astype(np.uint8)
        augmented_sample = (augmented_sample * 255).astype(np.uint8)
    print("Testing augmentation...")
    cv2.namedWindow("window_one", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("window_two", cv2.WINDOW_AUTOSIZE)
    for i in range(0, augmented_sample.shape[0]):
        original_x = cv2.resize(sample_x[i], (700, 700))
        augmented_x = cv2.resize(augmented_sample[i], (700, 700))
        cv2.imshow("window_one", original_x)
        cv2.imshow("window_two", augmented_x)
        cv2.waitKey(0)


main()

