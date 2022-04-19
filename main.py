import cv2
import numpy as np
import math
import sys
from shapely.geometry import LineString
from helpers import *
from core_functions import *

def main():
    if(len(sys.argv) < 2):
        return
    img = None
    if(sys.argv[1] == '-c'):
        #print('here')
        img =  cv2.imread('./cropped_images/' + sys.argv[2] + '.jpg', -1)
    else:
        img =  cv2.imread('./blueprint_images/' + sys.argv[1] + '.jpg', -1)

    img = image_resize(img, height=1024)
    cv2.imshow("image",img)
    cv2.waitKey(0)

    edges, gray = getEdges(img, 100, 100)
    cv2.imshow("image",edges)
    cv2.waitKey(0)
    
    lines_image, endpoints, intersections = findLines(edges, 30, 10, 5)
    #print(endpoints)
    #print(intersections)
    endpoint_image = paintPoints(edges, endpoints)
    intersections_image = paintPoints(edges, intersections)
    cv2.imshow("image", lines_image)
    cv2.waitKey(0)
    cv2.imshow("image", endpoint_image)
    cv2.waitKey(0)
    cv2.imshow("image", intersections_image)
    cv2.waitKey(0)
    contour_image = seeContours(lines_image)
    #convex_hull_image = getConvex(lines_image, endpoints + intersections)
    # Display the final convex hull image
    cv2.imshow('ConvexHull', contour_image)
    cv2.waitKey(0)
    #cv2.imshow('ConvexHull', convex_hull_image)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()