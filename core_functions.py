import cv2
from cv2 import convexHull
import numpy as np
import math
import sys
from shapely.geometry import LineString
from helpers import *

def getEdges(src, thresh1, thresh2):
    img=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    gray = img
    edges = cv2.Canny(img, thresh1, thresh2)
    return edges, gray

def findLines(src, maxLineGap, minLineLength, extraLength):
    blank_image = create_blank(src.shape[1], src.shape[0], (255,255,255))
    lines = cv2.HoughLinesP(src, 1, np.pi/360, 115, maxLineGap=maxLineGap, minLineLength=minLineLength)
    
    #lines_drawn = []
    horizontal_lines = []
    vertical_lines = []
    endpoints = []
    intersections = []

    for line in lines:
        #print(line)
        x1,y1,x2,y2 = line[0]
        #check for precision instead of ==
        if(y1 == y2):
            horizontal_lines.append([x1-extraLength, y1, x2+extraLength, y2])
            endpoints.append([x1,y1])
            endpoints.append([x2,y2])
        if(x1 == x2):
            vertical_lines.append([x1, y1+extraLength, x2, y2-extraLength])
            endpoints.append([x1,y1])
            endpoints.append([x2,y2])
    #try check for 2 intersection
    #Convert into one for loop
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
        if(intersects):
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,0,0),3) 

    for line in vertical_lines:
        x1,y1,x2,y2 = line
        intersects = False
        for h_line in horizontal_lines:
            x3,y3,x4,y4 = h_line
            if(y1 > y3 and y2 < y3 and x1 >= x3 and x1 <= x4):
                intersects = True
                intersection_point = line_intersection([[x1,y1], [x2,y2]], [[x3,y3], [x4,y4]])
                if(intersection_point not in intersections):
                    intersections.append(intersection_point)
        if(intersects):
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,0,0),3)
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
        print(point[0][0])
        adjustedHull.append([math.floor(point[0][0]), math.floor(point[0][1])])
    cv2.drawContours(convex_hull_image, adjustedHull, -1, (255, 0, 0), 2)
    # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # edges_hull = cv2.Canny(gray, 100, 100)

    # convex_hull_image = create_blank(src.shape[1], src.shape[0], (255,255,255))
    # contours, hierarchy = cv2.findContours(edges_hull, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # #sort countours by perimeter
    # contours = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)
    # limit = 0
    # for contour in contours:
    #     #if(limit > 1):
    #     #    break
    #     convexHull = cv2.convexHull(contour)
    #     cv2.drawContours(convex_hull_image, [convexHull], -1, (255, 0, 0), 2)
    #     limit += 1
    return convex_hull_image

def seeContours(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    edges_hull = cv2.Canny(gray, 100, 100)

    contourImg = create_blank(src.shape[1], src.shape[0], (255,255,255))
    contours, hierarchy = cv2.findContours(edges_hull, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contourImg, contours, -1, (255, 0, 0), 2)
    return contourImg