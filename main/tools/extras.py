import cv2


def resize(img, factor):
    return cv2.resize(img, (0, 0), fx=factor, fy=factor)

