from collections import defaultdict
import numpy as np
import cv2
import os


class StochasticDataRetriever:
    """
    Used for sampling data from the data sets
    """
    def __init__(self, dir_to_class_labels, image_dims, grayscale=True, sample_evenly_across_classes=True,
                                                                    resize_images=False, load_in_memory=False):
        """
        Take the data directory and corresponding class label for each data class
        :param dir_to_class_labels: list of tuples, each a pair of data dir with corresponding class label
        :param image_dims: (height, width) tuple of ints with the dimensions of the images to retrieve
        :param grayscale: if True, single channel grayscale images will be used, otherwise 3-channel BGR images
        :param sample_evenly_across_classes: if True there would be an equal likelihood for sampling from each class,
        otherwise the likelihood will be scaled by the number of data points in the classes
        :param resize_images: if True, images will be resized to image_dims if necessary, else exception will be raised
        :param load_in_memory: will store data in memory if True, else it will read every time from disc
        """
        self.dir_to_class_labels = dir_to_class_labels
        self.sample_evenly = sample_evenly_across_classes
        self.load_in_memory = load_in_memory  # Whether or not to load in memory
        self.N = len(dir_to_class_labels)  # Number of classes
        self.data_size = 0  # Total number of data points across all classes
        self.dir_to_file_paths = defaultdict(list)  # Map of a dir path to a list of all file paths in it
        self.dir_to_files = defaultdict(list)  # Map of a dir path to a list of numpy arrays with the images in the dir
        self.class_label_to_dir = {}  # Map from a class label to corresponding directory
        self.image_dims = tuple(image_dims)
        self.grayscale = grayscale
        self.resize_images = resize_images

        # Perform some input checks
        present_labels = set()
        for class_dir, class_label in dir_to_class_labels:
            if os.path.isdir(class_dir) is not True:
                raise ValueError("Not a valid directory path: {}".format(class_dir))
            if isinstance(class_label, int) is not True:
                raise ValueError("Class label not an integer: {}".format(class_label))
            if class_label >= self.N:
                raise ValueError("Invalid class label: {}".format(class_label))
            if class_label in present_labels:
                raise ValueError("Non-unique class label: {}".format(class_label))
            present_labels.add(class_label)

        # Read data files in the directories
        for class_dir, class_label in dir_to_class_labels:
            self.class_label_to_dir[class_label] = class_dir
            files_in_dir = [os.path.join(class_dir, file) for
                            file in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, file))]
            self.data_size += len(files_in_dir)  # Add the number of files in the current class directory

            # Now process every file in the class_dir directory
            for file_in_dir in files_in_dir:
                self.dir_to_file_paths[class_dir].append(file_in_dir)  # Add the file to the map
                if load_in_memory:
                    image = None
                    if grayscale:
                        image = cv2.imread(file_in_dir, cv2.IMREAD_GRAYSCALE)  # Read file as a grayscale image
                        if image is not None:
                            image = image.reshape(image.shape + (1,))  # Add additional channel
                    else:
                        image = cv2.imread(file_in_dir, cv2.IMREAD_COLOR)  # Read file as 3-channel BGR image
                    if image is None:
                        raise ValueError("Invalid image: {}".format(file_in_dir))
                    if image.shape[0] != image_dims[0] or image.shape[1] != image_dims[1]:
                        if resize_images:
                            image = cv2.resize(image, (image_dims[1], image_dims[0]))
                            if grayscale:
                                image = image.reshape(image.shape + (1,))  # Add additional channel
                        else:
                            raise ValueError("Invalid image dimensions for: {}".format(file_in_dir))
                    self.dir_to_files[class_dir].append(image)  # Add numpy array to the map

    def sample_data(self, number_of_samples):
        """
        Generates samples from the classes
        :param number_of_samples: integer, number of samples to generate
        :return: a tuple of data points xs and corresponding class labels ys as one-hot vectors
        """
        channels = 1 if self.grayscale else 3
        sample_xs = np.zeros(shape=(number_of_samples, self.image_dims[0],
                                                self.image_dims[1], channels), dtype=np.uint8)
        sample_ys = np.zeros(shape=(number_of_samples, self.N), dtype=np.int16)

        # Now generate each sample
        for i in range(0, number_of_samples):
            if self.sample_evenly:
                sample_class = np.random.random_integers(0, self.N - 1)
            else:
                number_sample = np.random.random_integers(1, self.data_size)
                cumulative_sum = 0
                for class_dir, class_label in self.dir_to_class_labels:
                    class_weight = len(self.dir_to_file_paths[class_dir])  # Number of data points in class
                    cumulative_sum += class_weight
                    if cumulative_sum >= number_sample:
                        sample_class = class_label
                        break

            class_sample = self.sample_from_class(sample_class)  # Sample an image from the class
            sample_xs[i, :, :, :] = class_sample
            sample_ys[i, sample_class] = 1  # y_i is a one-hot vector

        return sample_xs, sample_ys

    def sample_from_class(self, class_label):
        """
        Generates a sample from the given class label
        :param class_label: an integer in [0, N) denoting the class label
        :return: a numpy array with the sample image
        """
        class_dir = self.class_label_to_dir[class_label]
        class_size = len(self.dir_to_file_paths[class_dir])  # Number of files in the class
        sample_index = np.random.randint(0, class_size)
        sample_file_path = self.dir_to_file_paths[class_dir][sample_index]
        sample_image = None
        if self.load_in_memory:
            sample_image = self.dir_to_files[class_dir][sample_index]
        elif self.grayscale:
            sample_image = cv2.imread(sample_file_path, cv2.IMREAD_GRAYSCALE)  # Read file as a grayscale image
            sample_image = sample_image.reshape(sample_image.shape+(1,))  # Add additional channel
        else:
            sample_image = cv2.imread(sample_file_path, cv2.IMREAD_COLOR)  # Read file as 3-channel BGR image
        if sample_image is None:
            raise ValueError("Invalid image: {}".format(sample_file_path))
        if sample_image.shape[0] != self.image_dims[0] or sample_image.shape[1] != self.image_dims[1]:
            if self.resize_images:
                sample_image = cv2.resize(sample_image, (self.image_dims[1], self.image_dims[0]))
                if self.grayscale:
                    sample_image = sample_image.reshape(sample_image.shape + (1,))  # Add additional channel
            else:
                raise ValueError("Invalid image dimensions for: {}".format(sample_file_path))

        return sample_image

    def get_data_shape(self):
        """
        Get the shape of a single data sample
        :return: a tuple with a sample's shape
        """
        channels = 1 if self.grayscale else 3
        return self.image_dims + (channels,)

    def get_number_of_classes(self):
        """
        Get the number of data classes
        :return: a non-negative integer
        """
        return self.N

    def get_image_channels(self):
        """
        Get the number of channels per image. 1 for grayscale, 3 for BGR
        :return: the number of color channels per image
        """
        channels = 1 if self.grayscale else 3
        return channels
