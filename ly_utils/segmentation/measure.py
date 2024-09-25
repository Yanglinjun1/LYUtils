##########################################################################################
# Description: Utility functions for measurement using AI segmentations.
##########################################################################################

import numpy as np

__all__ = [
    "clip_range",
    "calculate_circle",
    "fit_line",
    "points2line_distance",
    "line2line_distance",
    "calculate_angle_from_lines",
    "calculate_angle_from_points",
    "calculate_distances_between_two_point_sets",
]


def clip_range(xs, ys, size_x, size_y):
    """
    Simple function to plot line with the range restricted
    by the image size

    xs, ys: numpy arrays of size (n,)
    """
    preserved_index = np.bitwise_and(
        np.bitwise_and(xs > 0, xs < size_x), np.bitwise_and(ys > 0, ys < size_y)
    )

    return xs[preserved_index], ys[preserved_index]


def calculate_circle(point1, point2, point3):
    """
    Method implemented according to https://math.stackexchange.com/questions/213658/get-the-equation-of-a-circle-when-given-3-points

    A function to find the circle given 3 points on the circle
    each point is a numpy array of shape (2,)

    returns, y and x coordiantes and the radius of the circle
    """
    y1, x1 = point1
    y2, x2 = point2
    y3, x3 = point3

    A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    B = (
        (x1**2 + y1**2) * (y3 - y2)
        + (x2**2 + y2**2) * (y1 - y3)
        + (x3**2 + y3**2) * (y2 - y1)
    )
    C = (
        (x1**2 + y1**2) * (x2 - x3)
        + (x2**2 + y2**2) * (x3 - x1)
        + (x3**2 + y3**2) * (x1 - x2)
    )
    D = (
        (x1**2 + y1**2) * (x3 * y2 - x2 * y3)
        + (x2**2 + y2**2) * (x1 * y3 - x3 * y1)
        + (x3**2 + y3**2) * (x2 * y1 - x1 * y2)
    )

    center_x = -0.5 * B / A
    center_y = -0.5 * C / A
    radius = np.sqrt((B**2 + C**2 - 4.0 * A * D) / (4 * A**2))

    return center_y, center_x, radius


def fit_line(binary_mask, auto_correct=False, use_bbox=False, use_orientation=True):
    """
    Line fitting function for binary mask
    """

    if use_bbox:
        import cv2

        cnts, _ = cv2.findContours(
            binary_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        rect = cv2.minAreaRect(cnts[0])
        box = cv2.boxPoints(rect)

        dist1, dist2 = np.sqrt(np.sum(((box[3] - box[2]) ** 2))), np.sqrt(
            np.sum(((box[2] - box[1]) ** 2))
        )
        if dist1 > dist2:  # always consider the longer side (for now)
            slope = (box[3, 1] - box[2, 1]) / (box[3, 0] - box[2, 0] + 1e-8)
        else:
            slope = (box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0] + 1e-8)

        center_x, center_y = rect[0]
        intersect = center_y - slope * center_x

        return np.array([slope, intersect])

    if use_orientation:
        from skimage.measure import regionprops_table

        regionprops = regionprops_table(
            binary_mask.astype(np.uint8), properties=["orientation", "centroid"]
        )
        orientation = regionprops["orientation"]
        center_y, center_x = regionprops["centroid-0"], regionprops["centroid-1"]

        if orientation < 0:
            angle = -0.5 * np.pi - orientation
        else:
            angle = 0.5 * np.pi - orientation

        slope = np.tan(angle)
        intersect = center_y - slope * center_x

        return np.array([slope, intersect])

    if auto_correct:
        Ys, Xs = np.where(binary_mask)
        range_Y, range_X = Ys.max() - Ys.min(), Xs.max() - Xs.min()

        if range_X > range_Y:
            p = np.polyfit(Xs, Ys, deg=1)
        else:
            p_inverse = np.polyfit(Ys, Xs, deg=1)

            # transfer the fitted poly parameter back as the X -> Y
            p = np.array([1.0 / p_inverse[0], -1.0 * p_inverse[1] / p_inverse[0]])
    else:
        p = np.polyfit(Xs, Ys, deg=1)

    return p[..., np.newaxis]


def points2line_distance(Ys, Xs, p):
    distances = (Ys - p[0] * Xs - p[1]) / np.sqrt(1.0 + p[0] ** 2)

    return distances


def line2line_distance(p1, p2):
    assert (
        p1[0] == p2[0]
    ), "different slope values were noticed; two lines must be parallel to each other."

    intersect_diff = (
        p2[1] - p1[1]
    )  # using the line 1 as the reference; this makes it a vector (0, intersect_diff)
    vector_dot = (
        -1.0 * intersect_diff / (p1[0] + 1e-7)
        if p1[0] < 0
        else intersect_diff / (p1[0] + 1e-7)
    )
    normal_vector_mod = np.sqrt(1.0 + 1.0 / (p1[0] ** 2 + 1e-7))

    return (vector_dot / normal_vector_mod).item()  # signed distance


def calculate_angle_from_lines(p1, p2, degree=True):
    angle1 = np.arctan2(p1[0], 1.0)
    angle2 = np.arctan2(p2[0], 1.0)

    angle = max(angle1, angle2) - min(angle1, angle2)
    if degree:
        angle *= 57.29578  # radian to degree

    return angle


def calculate_angle_from_points(vertex=None, p1=None, p2=None, degree=True):
    """
    function to calculate the angle between two vectors defined by three points
    """
    if vertex is None or p1 is None or p2 is None:
        return None

    # using the formula |a|*|b|*cos(theta) = a * b
    vector1, vector2 = p1 - vertex, p2 - vertex
    angle = np.sum(vector1 * vector2) / (
        np.linalg.norm(vector1, ord=2) * np.linalg.norm(vector2, ord=2)
    )
    angle = np.arccos(angle)

    if degree:
        angle *= 57.29578  # radian to degree

    return angle


def calculate_distances_between_two_point_sets(set1, set2):
    """
    Calculate the Euclidean distances between each pair of points from two sets.

    Parameters:
    - set1 (np.array): An NxD array where N is the number of points and D is the dimension of each point.
    - set2 (np.array): An MxD array where M is the number of points and D is the dimension of each point.

    Returns:
    - np.array: An NxM array of distances where the element at (i, j) is the distance between
    set1[i] and set2[j].
    """
    # Ensure inputs are numpy arrays
    set1 = np.array(set1)
    set2 = np.array(set2)

    # Subtract each point in set1 from each point in set2
    # The shape of diff will be (N, M, D)
    diff = set1[:, np.newaxis, :] - set2[np.newaxis, :, :]

    # Compute the squared differences, sum across the columns, and take the square root
    distances = np.sqrt(np.sum(diff**2, axis=2))

    return distances
