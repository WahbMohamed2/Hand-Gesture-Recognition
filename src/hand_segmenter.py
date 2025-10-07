import cv2
import numpy as np


def segment_hand(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(contour)
        roi = frame[y : y + h, x : x + w]
        roi_resized = cv2.resize(roi, (120, 100))
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        gray = gray.reshape(1, 100, 120, 1)
        return gray, contour
    return None, None
