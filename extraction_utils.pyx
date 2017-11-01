cimport numpy as np  # For Cython
from collections import defaultdict
from functools import reduce
import numpy as np
import math
import time
import cv2
import os


def find_connected_components(binary_image):
    """
    Finds the connected components by assigning every pixel to a component ID
    :param binary_image: cv2 grayscale image
    :return: numpy matrix with the connected components
    """
    def find_connected_component(initial_coords, component_id):
        """
        Use BFS to mark all belonging pixels to the connected component
        :param initial_coords: (int, int) with a pixel belonging to the CC
        :param component_id: int, a unique ID number for the CC
        :return: void
        """
        queue = [initial_coords]
        marked[initial_coords[0], initial_coords[1]] = component_id
        while len(queue) > 0:
            current_coords = queue.pop(0) # Remove first element in the list
            row, col = current_coords
            for i in range(max(0, row-1), min(marked.shape[0], row+2)):
                for j in range(max(0, col-1), min(marked.shape[1], col+2)):
                    # Adjacent cell with coords (i, j)
                    if marked[i, j] == 0:
                        marked[i, j] = component_id
                        queue.append((i, j))

    cdef np.int16_t[:, :] marked = np.zeros(shape=binary_image.shape, dtype=np.int16)
    cdef np.uint8_t[:, :] reference_image = binary_image
    # Mark non-object pixels with -1
    for row in range(marked.shape[0]):
        for col in range(marked.shape[1]):
            if reference_image[row, col] == 0:
                marked[row, col] = -1

    # Now fill components
    connected_components = 0
    for row in range(marked.shape[0]):
        for col in range(marked.shape[1]):
            if marked[row, col] == 0:
                connected_components += 1
                find_connected_component((row, col), connected_components)

    return marked


def find_nearest_connected_component(int row, int col, np.int16_t[:, :] marked):
    """
    Using BFS to find the nearest connected component relative to the given coordinates
    :param row: the row coordinate of the pixel
    :param col: the column coordinate of the pixel
    :param marked: numpy matrix with IDs of the connected components
    :return: the id of the nearest connected component
    """
    nearest_cc = -1
    visited_pixels = set()
    visited_pixels.add((row, col))
    queue = [(row, col)]

    while len(queue) > 0:
        current_row, current_col = queue.pop(0)  # Popping first element in the list
        current_id = marked[current_row, current_col]
        if current_id != -1:
            nearest_cc = current_id
            break

        for i in range(max(0, current_row-1), min(marked.shape[0], current_row+2)):
            for j in range(max(0, current_col-1), min(marked.shape[1], current_col+2)):
                # Adjacent cell with coords (i, j)
                if (i, j) not in visited_pixels:
                    visited_pixels.add((i, j))
                    queue.append((i, j))

    return nearest_cc


def get_component_influence_map(np.int16_t[:, :] marked):
    """
    Assigns every void pixel to the closest connected component to it using BFS
    :param marked: numpy matrix with IDs of the connected components
    :return: a numpy matrix with CC ids
    """
    def is_border_pixel(int row, int col):
        """
        Checks if the pixel has a void pixel as a neighbour
        :param row: the row coordinate of the cell
        :param col: the column coordinate of the cell
        :return: boolean
        """
        for i in range(max(0, row-1), min(marked.shape[0], row+2)):
            for j in range(max(0, col-1), min(marked.shape[1], col+2)):
                # Adjacent cell with coords (i, j)
                if marked[i, j] == -1:
                    return True

        return False

    cdef np.int16_t[:, :] influence_map = np.copy(marked)
    cdef list queue = []
    # Add every border pixel to the queue
    for row in range(0, marked.shape[0]):
        for col in range(0, marked.shape[1]):
            if marked[row, col] != -1 and is_border_pixel(row, col):
                queue.append((row, col))

    while len(queue) > 0:
        row, col = queue.pop(0)  # Pop the first element from the list
        for i in range(max(0, row-1), min(marked.shape[0], row+2)):
            for j in range(max(0, col-1), min(marked.shape[1], col+2)):
                # Adjacent cell with coords (i, j)
                if influence_map[i, j] == -1:
                    influence_map[i, j] = influence_map[row, col]
                    queue.append((i, j))

    return influence_map


def get_image_objects(np.int16_t[:, :] marked, min_object_size=20):
    """
    Generates the center coordinates of the objects in the binary image
    Uses a Breadth-First Search flood fill to find the connected components
    :param marked: numpy matrix with IDs of the connected components
    :param min_object_size: minimum number of pixels in an object to be considered as such
    :return: tuple of dict with cc id keys and (int, int) values with the coordinates of the centers of mass of
    the objects. First tuple value is for the size filtered CCs, second is for all of them
    """
    # Now find components' centers of mass
    component_size = defaultdict(int)  # Keeps number of pixels in each component
    center_x = {}
    center_y = {}
    cumulative_x = defaultdict(int)
    cumulative_y = defaultdict(int)
    # Find component sizes
    for row in range(0, marked.shape[0]):
        for col in range(0, marked.shape[1]):
            if marked[row, col] != -1:
                cc_id = marked[row, col]
                component_size[cc_id] += 1

    # Find y axis
    for row in range(0, marked.shape[0]):
        for col in range(0, marked.shape[1]):
            if marked[row, col] != -1:
                cc_id = marked[row, col]
                cumulative_y[cc_id] += 1
                if math.fabs(component_size[cc_id]/2 - cumulative_y[cc_id]) < 2:
                    center_y[cc_id] = row
    # Find x axis
    for col in range(0, marked.shape[1]):
        for row in range(0, marked.shape[0]):
            if marked[row, col] != -1:
                cc_id = marked[row, col]
                cumulative_x[cc_id] += 1
                if math.fabs(component_size[cc_id]/2 - cumulative_x[cc_id]) < 2:
                    center_x[cc_id] = col

    filtered_cc_center_coords = {}
    all_cc_center_coords = {}
    for cc_id in component_size:
        component_coords = center_y[cc_id], center_x[cc_id]
        all_cc_center_coords[cc_id] = component_coords
        if component_size[cc_id] > min_object_size:
            filtered_cc_center_coords[cc_id] = component_coords

    return filtered_cc_center_coords, all_cc_center_coords


def find_void_intensity(grayscale_image):
    """
    Computes the pixel intensity of the empty space in the image
    :param grayscale_image: cv2 grayscale image
    :return: an integer with the computed intensity
    """
    def rec_find_intensity(grayscale_image, min_intensity, max_intensity):
        """
        A recursive auxiliary function. With every call the range of possible intensities is narrowed by
        computing the histogram of intensities in the given range and picking the one with highest pixel count
        :param grayscale_image: cv2 grayscale image
        :param min_intensity: number
        :param max_intensity: number
        :return: an integer with the computed intensity
        """
        if round(min_intensity) == round(max_intensity):
            return min_intensity

        histogram_bins = 2
        histogram, bin_edges = np.histogram(grayscale_image, bins=histogram_bins, range=(min_intensity, max_intensity))
        max_bin = 0
        max_val = 0
        for idx, hist_val in enumerate(histogram):
            if hist_val > max_val:
                max_val = hist_val
                max_bin = idx
        values_per_bin = (max_intensity-min_intensity+1) / histogram_bins
        new_min_intensity = min_intensity + max_bin*values_per_bin
        new_max_intensity = new_min_intensity + values_per_bin-1
        return rec_find_intensity(grayscale_image, new_min_intensity, new_max_intensity)

    highest_possible = 63 # Max possible void intensity
    return rec_find_intensity(grayscale_image, 0, highest_possible)


def get_component_bounding_boxes(np.int16_t[:, :] marked):
    """
    Computes the coordinates of the upper left and bottom right corners of the
    bounding box wrapped around every connected component
    :param marked: numpy matrix filled with the CC IDs for every pixel
    :return: a dict with a key CC id and value a 2-tuple with coords for the corners
    """
    bounding_boxes = {}
    for row in range(0, marked.shape[0]):
        for col in range(0, marked.shape[1]):
            cc_id = marked[row, col]
            if cc_id == -1:
                continue
            if cc_id in bounding_boxes:
                bounding_box = bounding_boxes[cc_id]
                upper_left = min(bounding_box[0][0], row), min(bounding_box[0][1], col)
                bottom_right = max(bounding_box[1][0], row), max(bounding_box[1][1], col)
                bounding_boxes[cc_id] = upper_left, bottom_right
            else:
                bounding_box = (row, col), (row, col)
                bounding_boxes[cc_id] = bounding_box

    return bounding_boxes


def near_side_particle(int row, int col, int check_within_dist, int center_id, np.int16_t[:, :] marked):
    """
    Checks if the cell at the given coordinates is close enough to a side particle
    :param row: the row coordinate of the cell
    :param col: the column coordinate of the cell
    :param check_within_dist: number of pixels away to check for adjacency
    :param center_id: id of the connected component in the center of the crop
    :param marked: numpy matrix filled with the CC IDs for every pixel
    :return: boolean, True if close enough, False otherwise
    """
    for i in range(max(0, row-check_within_dist), min(row+check_within_dist+1, marked.shape[0])):
        for j in range(max(0, col-check_within_dist), min(col+check_within_dist+1, marked.shape[1])):
            if marked[i, j] != -1 and marked[i, j] != center_id:
                return True

    return False


def near_center_particle(int row, int col, int check_within_dist, int center_id, np.int16_t[:, :] marked):
    """
    Checks if the cell at the given coordinates is close enough to the central particle
    :param row: the row coordinate of the cell
    :param col: the column coordinate of the cell
    :param check_within_dist: number of pixels away to check for adjacency
    :param center_id: id of the connected component in the center of the crop
    :param marked: numpy matrix filled with the CC IDs for every pixel
    :return: boolean, True if close enough, False otherwise
    """
    for i in range(max(0, row-check_within_dist), min(row+check_within_dist+1, marked.shape[0])):
        for j in range(max(0, col-check_within_dist), min(col+check_within_dist+1, marked.shape[1])):
            if marked[i, j] == center_id:
                return True

    return False

def average_void_intensity_at_pixel(int row, int col, np.uint8_t[:, :] grayscale_image, np.int16_t[:, :] marked):

    # List of intensities definitely not part of particle
    intensities = []
    for row in range(max(0, row-20), min(row+20, grayscale_image.shape[0])):
        for col in range(max(0, col-20), min(col+20, grayscale_image.shape[1])):
            if marked[row, col] == -1 and not near_side_particle(row, col, 14, -1, marked):
                pixel_intensity = grayscale_image[row, col]
                intensities.append(pixel_intensity)

    return find_void_intensity(intensities)


def remove_side_objects(grayscale_crop, np.int16_t[:, :] marked, center_id, average_void_intensity):
    """
    Removes all objects that are not the center one
    :param grayscale_crop: cv2 grayscale image
    :param marked: numpy matrix filled with the CC IDs for every pixel
    :param center_id: id of the connected component in the center of the crop
    :param average_void_intensity: dict of cc id keys and average void intensity around that component
    :return: cv2 grayscale image
    """
    center_y, center_x = int(marked.shape[0]/2), int(marked.shape[1]/2)

    # Now replace side objects with the average void
    cc_influence_map = get_component_influence_map(marked)
    gaussian_variance = 0.75
    dist_threshold = 12
    cleaned_crop = grayscale_crop.copy()
    for row in range(grayscale_crop.shape[0]):
        for col in range(grayscale_crop.shape[1]):
            if marked[row, col] != -1 and marked[row, col] != center_id:
                cc_id = marked[row, col]
                cc_void = average_void_intensity[cc_id]
                cleaned_crop[row, col] = np.random.normal(cc_void, gaussian_variance)  # Sample from a Gaussian
            # Apply heuristic to remove ambiguous contours
            elif marked[row, col] == -1 and near_side_particle(row, col, dist_threshold, center_id, marked) and \
                                                 not near_center_particle(row, col, dist_threshold, center_id, marked):
                #nearest_cc_id = find_nearest_connected_component(row, col, marked)
                nearest_cc_id = cc_influence_map[row, col]
                cc_void = average_void_intensity[nearest_cc_id]
                #cc_void = average_void_intensity_at_pixel(row, col, grayscale_crop, marked)
                cleaned_crop[row, col] = np.random.normal(cc_void, gaussian_variance)  # Sample from a Gaussian

    # Now smoothen edges of removed side objects
    erasure_dist_threshold = 18
    blurred_image = cv2.GaussianBlur(cleaned_crop, ksize=(7, 7), sigmaX=2)
    for row in range(grayscale_crop.shape[0]):
        for col in range(grayscale_crop.shape[1]):
            # Now blur removed objects
            if marked[row, col] == -1 and near_side_particle(row, col, erasure_dist_threshold, center_id, marked) and \
                                                not near_center_particle(row, col, dist_threshold, center_id, marked):
                mu = blurred_image[row, col]
                cleaned_crop[row, col] = np.random.normal(mu, gaussian_variance)  # Sample from a Gaussian

    return cleaned_crop
