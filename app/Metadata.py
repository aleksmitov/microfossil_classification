import json


class Metadata:
    def __init__(self, metadata_file):
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

        self.models_dir = models_dir
        self.number_of_classes = number_of_classes
        self.input_image_dims = input_image_dims