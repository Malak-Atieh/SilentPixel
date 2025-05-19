import cv2
import numpy as np
import io

def detect_busy_areas(image_bytes, sensitivity='medium'):
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    thresholds = {'low': 15, 'medium': 25, 'high': 40}
    thresh_val = thresholds.get(sensitivity, 25)
    mask = np.uint8(np.abs(laplacian) > thresh_val)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100: 
            areas.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)})
    return areas
