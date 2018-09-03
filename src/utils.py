import cv2
import numpy as np

def process_image_with_resize_and_grayed(image):
    resized = cv2.resize(image, (80, 80), interpolation=cv2.INTER_CUBIC)
    grayed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, thsh_img = cv2.threshold(grayed,1,255,cv2.THRESH_BINARY)
    processed_img = np.reshape(thsh_img, (80, 80, 1))
    return processed_img