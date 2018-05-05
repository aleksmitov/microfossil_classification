import pyximport; pyximport.install()
from extraction_utils import find_connected_components, get_image_objects, remove_side_objects, \
    get_component_bounding_boxes, find_void_intensity, near_side_particle, near_center_particle
from functools import reduce
import numpy as np
import cv2
import os
import time


MIN_MICROFOSSIL_SIZE = 300  # Number of pixels in an object to be considered a microfossil
CROP_DIMS = (200, 200)  # The dimensions of every crop around a microfossil
REMOVE_SIDE_PARTICLES = True  # If true, particles other than the central will be removed
SOURCE_DIR = "./raw_images"
DESTINATION_DIR = "./raw_images_processed"


def get_binary_image(grayscale_image):
    """
    Convert to a binary image with either 0 or 255 intensities
    Use Otsu's method for thresholding
    :param grayscale_image: cv2 grayscale image
    :return: cv2 grayscale image
    """
    _, thresholded_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_image


def filter_crop(grayscale_image):
    """
    Removes the side particles for the given crop of a microfossil
    :param grayscale_image: cv2 grayscale image
    :return: cv2 grayscale image/numpy array
    """
    # Blurring the image helps with getting a more consistent binary image
    blurred_image = cv2.bilateralFilter(grayscale_image, d=0, sigmaColor=40, sigmaSpace=2)
    binary_image = get_binary_image(blurred_image)
    marked = find_connected_components(binary_image)
    _, all_coords = get_image_objects(marked, 0)
    M, N = grayscale_image.shape
    average_void_intensity = compute_average_void_intensity(grayscale_image, marked, all_coords)
    cc_id = -1
    # Finding the cc id of the centered particle
    for i in range(N/2, -1, -1):
        current_cc = marked[M/2, i]
        if current_cc != -1:
            cc_id = current_cc
            break

    filtered_crop = remove_side_objects(grayscale_image, marked, cc_id, average_void_intensity)

    return filtered_crop


def filter_crops_in_dir(source_dir, destination_dir):
    """
    Recursively extracts microfossils from images in the source dir and its subdirs
    :param source_dir: string with the path to the source folder
    :param destination_dir: string with the path to the destination folder
    :return: void
    """
    if os.path.isdir(source_dir) is False:
        raise Exception("Not a valid source path")
    if os.path.isdir(destination_dir) is False:
        os.makedirs(destination_dir)

    print("Currently processing images in dir: {}".format(source_dir))
    image_extensions = [".tif", ".TIF", ".png", ".PNG"]
    sub_dirs = []
    images_in_dir = []
    for file in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, file)) and os.path.join(source_dir, file) != destination_dir:
            sub_dirs.append(file)
        # If it's an image with the given extensions
        elif reduce((lambda x, y: x or y), [file.endswith(ext) for ext in image_extensions]):
            images_in_dir.append(file)

    # Now process the images
    for image_path in images_in_dir:
        full_image_path = os.path.join(source_dir, image_path)
        grayscale_image = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
        if grayscale_image is None:
            print("Couldn't read image and was skipped: {}".format(full_image_path))
            continue

        filtered_crop = filter_crop(grayscale_image)
        if np.array_equal(grayscale_image, filtered_crop) is False:
            filtered_crop_file_name = "{}_filtered.tif".format(os.path.splitext(image_path)[0])
            cv2.imwrite(os.path.join(destination_dir, filtered_crop_file_name), filtered_crop)
        unfiltered_crop_file_name = "{}_unfiltered.tif".format(os.path.splitext(image_path)[0])
        cv2.imwrite(os.path.join(destination_dir, unfiltered_crop_file_name), grayscale_image)

    # Recursively apply to all subdirs
    for subdir in sub_dirs:
        source_subdir = os.path.join(source_dir, subdir)
        destination_subdir = os.path.join(destination_dir, subdir)
        filter_crops_in_dir(source_subdir, destination_subdir)


def compute_average_void_intensity(grayscale_image, marked, particle_coords):
    """
    Computing the void intensity around the connected components
    :param grayscale_image: cv2 grayscale image
    :param marked: numpy matrix with IDs of the connected components
    :param particle_coords: dictionary with (cc_id, (cc_center_y, cc_center_x)) pairs
    :return: dictionary with (cc_id, void_intensity) pairs
    """
    M, N = grayscale_image.shape
    average_void_intensity = {}
    bounding_boxes = get_component_bounding_boxes(marked)
    void_crop_increase = 22
    for cc_id in particle_coords:
        upper_left = max(bounding_boxes[cc_id][0][0]-void_crop_increase, 0),\
                                                        max(bounding_boxes[cc_id][0][1]-void_crop_increase, 0)
        bottom_right = min(bounding_boxes[cc_id][1][0]+void_crop_increase, M-1),\
                                                        min(bounding_boxes[cc_id][1][1]+void_crop_increase, N-1)
        cc_void_crop = grayscale_image[upper_left[0]:bottom_right[0], upper_left[1]:bottom_right[1]]
        marked_void_crop = marked[upper_left[0]:bottom_right[0], upper_left[1]:bottom_right[1]]

        # List of intensities definitely not part of particle
        intensities = []
        for row in range(0, cc_void_crop.shape[0]):
            for col in range(0, cc_void_crop.shape[1]):
                if marked_void_crop[row, col] == -1 and not near_side_particle(row, col, 14, cc_id, marked_void_crop) \
                                                    and not near_center_particle(row, col, 14, cc_id, marked_void_crop):
                    pixel_intensity = cc_void_crop[row, col]
                    intensities.append(pixel_intensity)

        average_void_intensity[cc_id] = find_void_intensity(intensities)

    return average_void_intensity


def extract_microfossils(grayscale_image, min_microfossil_pixel_size, crop_dims, remove_side_particles):
    """
    Extracts all microfossil particles from the image
    :param grayscale_image: cv2 grayscale image
    :param min_microfossil_pixel_size: filter out objects with smaller pixel size
    :param crop_dims: the dimensions of the crops
    :param remove_side_particles: boolean, if True particles except for the central are removed
    :return: a tuple of lists of the unfiltered and filtered crops
    """
    # Blurring the image helps with getting a more consistent binary image
    blurred_image = cv2.bilateralFilter(grayscale_image, d=0, sigmaColor=40, sigmaSpace=2)
    binary_image = get_binary_image(blurred_image)
    marked = find_connected_components(binary_image)
    coords, all_coords = get_image_objects(marked, min_microfossil_pixel_size)
    M, N = grayscale_image.shape

    # Computing the void intensity around the connected components
    average_void_intensity = compute_average_void_intensity(grayscale_image, marked, all_coords)

    # Getting the crops
    filtered_crops, unfiltered_crops = [], []
    for cc_id in coords:
        obj_row, obj_col = coords[cc_id]
        from_x = int(obj_col - crop_dims[1] / 2)
        from_y = int(obj_row - crop_dims[0] / 2)
        valid_y = from_y >= 0 and from_y + crop_dims[0] < M
        valid_x = from_x >= 0 and from_x + crop_dims[1] < N
        if valid_x and valid_y:
            crop_img = grayscale_image[from_y:from_y+crop_dims[0], from_x:from_x+crop_dims[1]]
            unfiltered_crops.append(crop_img)
            if remove_side_particles:
                crop_cc = marked[from_y:from_y+crop_dims[0], from_x:from_x+crop_dims[1]]
                filtered_crop = remove_side_objects(crop_img, crop_cc, cc_id, average_void_intensity)
                filtered_crops.append(filtered_crop)

    return unfiltered_crops, filtered_crops


def extract_microfossils_in_dir(source_dir, destination_dir,
                                            crop_dims, min_microfossil_size, clean_particles):
    """
    Recursively extracts microfossils from images in the source dir and its subdirs
    :param source_dir: string with the path to the source folder
    :param destination_dir: string with the path to the destination folder
    :param crop_dims: tuple of (height, width) of crops generated
    :param min_microfossil_size: min size of CC for generated crops
    :param clean_particles: boolean
    :return: void
    """
    if os.path.isdir(source_dir) is False:
        raise Exception("Not a valid source path")
    if os.path.isdir(destination_dir) is False:
        os.makedirs(destination_dir)

    print("Currently processing images in dir: {}".format(source_dir))
    image_extensions = [".tif", ".TIF", ".png", ".PNG"]
    sub_dirs = []
    images_in_dir = []
    for file in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, file)) and os.path.join(source_dir, file) != destination_dir:
            sub_dirs.append(file)
        # If it's an image with the given extensions
        elif reduce((lambda x, y: x or y), [file.endswith(ext) for ext in image_extensions]):
            images_in_dir.append(file)

    # Now process the images
    for image_path in images_in_dir:
        full_image_path = os.path.join(source_dir, image_path)
        grayscale_image = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
        if grayscale_image is None:
            print("Couldn't read image and was skipped: {}".format(full_image_path))
            continue

        unfiltered_crops, filtered_crops = extract_microfossils(grayscale_image, min_microfossil_size,
                                                                crop_dims, clean_particles)
        for idx, crop in enumerate(unfiltered_crops):
            crop_file_name = "{}_crop_{}_unfiltered.png".format(os.path.splitext(image_path)[0], idx)
            cv2.imwrite(os.path.join(destination_dir, crop_file_name), crop)
        for idx, crop in enumerate(filtered_crops):
            crop_file_name = "{}_crop_{}_filtered.png".format(os.path.splitext(image_path)[0], idx)
            cv2.imwrite(os.path.join(destination_dir, crop_file_name), crop)

    # Recursively apply to all subdirs
    for subdir in sub_dirs:
        source_subdir = os.path.join(source_dir, subdir)
        destination_subdir = os.path.join(destination_dir, subdir)
        extract_microfossils_in_dir(source_subdir, destination_subdir,
                                                    crop_dims, min_microfossil_size, clean_particles)


#start_time = time.time()
#extract_microfossils_in_dir(SOURCE_DIR, DESTINATION_DIR)
#filter_crops_in_dir(SOURCE_DIR, DESTINATION_DIR)
#elapsed_time = time.time() - start_time
#print("Done extracting microfossils. Time spent: {}sec.".format(elapsed_time))