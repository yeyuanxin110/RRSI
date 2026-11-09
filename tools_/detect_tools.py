"""Single-scale gradient FAST keypoint detection."""

import cv2 as cv
import numpy as np


def float2uint8(values):
    maximum = np.max(values)
    minimum = np.min(values)
    if maximum == minimum:
        return np.zeros(values.shape, dtype=np.uint8)
    return ((values - minimum) / (maximum - minimum) * 255).astype(np.uint8)


def FAST_kps_detect(image, num_max=7500):
    if num_max < 1:
        raise ValueError("num_max must be positive.")
    threshold = 0
    while True:
        detector = cv.FastFeatureDetector_create(threshold)
        keypoints = detector.detect(image, None)
        if not keypoints:
            return np.empty((0, 2), dtype=np.float32)
        points = cv.KeyPoint_convert(keypoints)
        threshold += 5
        if len(points) < num_max:
            return points


def gradient(image):
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=1, scale=0.5, delta=0)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=1, scale=0.5, delta=0)
    return np.sqrt(np.multiply(sobel_x, sobel_x) + np.multiply(sobel_y, sobel_y))


def FAST_gradient_kps_detect(image, num_max=7500):
    return FAST_kps_detect(float2uint8(gradient(image)), num_max=num_max)


def keypoint_detect(image, num_intervals=0, num_max=7500, max_scale=1.0):
    if num_intervals != 0:
        raise ValueError("This inference package uses single-scale detection.")
    keypoints = FAST_gradient_kps_detect(image, num_max=num_max)
    return [image], [keypoints], [1.0], len(keypoints)
