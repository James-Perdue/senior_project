from pdf2image import convert_from_path
import cv2
import numpy as np
import math
import sys
from shapely.geometry import LineString
from PyQt5.QtWidgets import QApplication
def getDPI():
    app = QApplication(sys.argv)
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    app.quit()
    return dpi

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def line_length(line):
    x1, y1, x2, y2 = line
    
    return math.sqrt((x2-x1)**2 + (y2-y1) ** 2)

def same_direction(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    if(x1 - x2 == x3 - x4 or y1 - y2 == y3 - y4):
        return True
    return False

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [math.floor(x), math.floor(y)]

def getRectCenters(rects):
    centers = []
    for rect in rects:
        x1, y1, x2, y2 = rect
        centers.append([(x1 + x2) // 2, (y1 + y2) // 2])
    return centers