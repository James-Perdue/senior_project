import cv2
import numpy as np
import math
from helpers import *

#This method was scrapped as it was a hack to find labels by color of the text
def getPossibleTitles(src):
    hsv=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    # Threshold of blue in HSV space
    lower_blue = np.array([140, 130, 200])
    upper_blue = np.array([180, 200, 255])
    
    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = src.copy()
    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    dilate = cv2.dilate(mask, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rectangles = []
    for c in cnts:
        area = cv2.contourArea(c)
        #print(area)
        if area > 2000 and area < 5000:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (36,255,12), 3)
            rectangles.append([x, y, x + w, y + h])
    return result, rectangles

def drawRect(src, rectangles):
    result = src.copy()
    for rect in rectangles:
        x, y, x2, y2 = rect
        cv2.rectangle(result, (x, y), (x2, y2), (36,255,12), 3)
    return result


def getEdges(src, thresh1, thresh2):
    img=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    gray = img
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)
    edges = cv2.Canny(image_sharp, thresh1, thresh2)
    return edges, gray

def findLines(src, maxLineGap, minLineLength, extraLength, thickness = 1, lineThreshold=150):
    blank_image = create_blank(src.shape[1], src.shape[0], (255,255,255))
    lines = cv2.HoughLinesP(src, 1, np.pi/360, lineThreshold, maxLineGap=maxLineGap, minLineLength=minLineLength)

    horizontal_lines = []
    vertical_lines = []
    endpoints = []
    intersections = []

    for line in lines:
        x1,y1,x2,y2 = line[0]
        #I check for precision instead of exact horizontal or vertical
        if(abs(y1 - y2) < 3):
            horizontal_lines.append([x1-extraLength, y1, x2+extraLength, y2])
            endpoints.append([x1,y1])
            endpoints.append([x2,y2])
        if(abs(x1 - x2) < 3):
            vertical_lines.append([x1, y1+extraLength, x2, y2-extraLength])
            endpoints.append([x1,y1])
            endpoints.append([x2,y2])
            
    for line in horizontal_lines:
        x1,y1,x2,y2 = line
        intersects = False

        for v_line in vertical_lines:
            x3,y3,x4,y4 = v_line
            if(x1 < x3 and x2 > x3 and y1 <= y3 and y1 >= y4):
                intersects = True
                intersection_point = line_intersection([[x1,y1], [x2,y2]], [[x3,y3], [x4,y4]])
                if(intersection_point not in intersections):
                    intersections.append(intersection_point)
                cv2.line(blank_image,(x3,y3),(x4,y4),(0,0,0), thickness)
        if(intersects):
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,0,0), thickness) 
    return blank_image, endpoints, intersections

def paintPoints(src, points):
    point_image = create_blank(src.shape[1], src.shape[0], (255,255,255))
    for point in points:
        x, y = point
        cv2.circle(point_image, (x,y), radius=2, color=(0, 0, 0), thickness=-1)
    return point_image

def getConvex(src, points):
    convex_hull_image = create_blank(src.shape[1], src.shape[0], (255,255,255))
    convexHull = cv2.convexHull(np.array(points, dtype='float32'))
    adjustedHull = []
    for point in convexHull:
        adjustedHull.append([math.floor(point[0][0]), math.floor(point[0][1])])
    cv2.drawContours(convex_hull_image, adjustedHull, -1, (255, 0, 0), 2)
    return convex_hull_image

def getContours(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    edges_hull = cv2.Canny(gray, 100, 100)

    contourImg = create_blank(src.shape[1], src.shape[0], (255,255,255))
    contours, hierarchy = cv2.findContours(edges_hull, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contourImg, contours, -1, (255, 0, 0), 2)
    return contourImg

def fillRooms(src, centers, colors):
    result = src.copy()
    
    color_index = 0
    for center in centers:
        blue = colors[color_index][0]
        green = colors[color_index][1]
        red = colors[color_index][2]
        cv2.floodFill(result, None, seedPoint=center, newVal=(blue, green, red), loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
        if(color_index ==len(colors) -1):
            color_index = 0
        else:
            color_index += 1
    return result

def getArea(src, b, g, r):
    COL_MIN = np.array([b, g, r], np.uint8)
    # maximum value of pixels in BGR order -> given bgr params
    COL_MAX = np.array([b, g, r], np.uint8)

    dst = cv2.inRange(src, COL_MIN, COL_MAX)
    num_red = cv2.countNonZero(dst)

    return(num_red)
    
def convertPixeltoSquareFeet(pixels, DPI, scaleFactor):
    squareInch = DPI**2
    return (pixels / squareInch) / float(scaleFactor) ** 2

def mergeImages(base, overlay, alpha):
    res = cv2.addWeighted(base, .7, overlay, alpha, 0)
    return res