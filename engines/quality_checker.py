import numpy as np
import cv2
import math

def quality_assessment(mask):
    contour = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    if perimeter > 0:
        circularity = (4 * math.pi * area) / (perimeter ** 2)
    else:
        circularity = 0

    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, orientation) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
    else:
        eccentricity = 0

    return {
        'area': float(area),
        'perimeter': float(perimeter),
        'circularity': float(circularity),
        'eccentricity': float(eccentricity)
    }