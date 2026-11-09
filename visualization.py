"""Draw connecting lines between paired keypoints."""

import cv2 as cv


def draw_matches(keypoints1, keypoints2, matches, image1, image2):
    kps1 = [cv.KeyPoint(p[0], p[1], 1) for p in keypoints1]
    kps2 = [cv.KeyPoint(p[0], p[1], 1) for p in keypoints2]
    return cv.drawMatches(image1, kps1, image2, kps2, matches,
                           None, flags=2, matchColor=[0,255,255])
