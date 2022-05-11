from textwrap import fill
import cv2
from cv2 import WINDOW_AUTOSIZE
from cv2 import WINDOW_GUI_NORMAL
import numpy as np
import easygui
import unicodedata
import sys
from helpers import *
from core_functions import *

def callback(x):
    pass

def main():
    uni_img = easygui.fileopenbox()
    
    scale_factor = "test"
    scale_factor = easygui.enterbox("Enter Scale Factor", "Scale Factor", "")
    img = cv2.imread(uni_img, 1)

    img = image_resize(img, height=1024)
    dpi = getDPI()
    print(dpi)
    #This fn is very specific to 2bedhouse1 due to color thresholding
    #blue, selection_squares = getPossibleTitles(img)
    edges, gray = getEdges(img, 100, 100)
    
    lines_image, endpoints, intersections = findLines(edges, maxLineGap=30, minLineLength=10, extraLength=2, thickness=1, lineThreshold=180)
    blue_gone = findNonBlue(edges, img, 30, 10, 5, 1)
    endpoint_image = paintPoints(edges, endpoints)
    intersections_image = paintPoints(edges, intersections)

    contour_image = getContours(lines_image)
    #convex_hull_image = getConvex(lines_image, endpoints + intersections)
    # Display the final convex hull image
    #Check regions first
    # window_name = "Takeoff Software: Scale Factor = " + str(scale_factor)
    # cv2.namedWindow(window_name, WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL)
    # cv2.createTrackbar("Image", window_name, 0, 7, callback)
    # while True:
    #     # get Trackbar position
    #     pos = cv2.getTrackbarPos("Image", window_name)
    #     if pos == 0:
    #         cv2.imshow(window_name, img)
    #     elif pos == 1:
    #         cv2.imshow(window_name, edges)
    #     elif pos == 2:
    #         cv2.imshow(window_name, lines_image)
    #     elif pos == 3:
    #         cv2.imshow(window_name, blue_gone)
    #     elif pos == 4:
    #         cv2.imshow(window_name, endpoint_image)
    #     elif pos == 5:
    #         cv2.imshow(window_name, contour_image)
    #     elif pos == 6:
    #         cv2.imshow(window_name, blue)
    #     # elif pos == 7:
    #     #     cv2.imshow(window_name, text)
    #     key = cv2.waitKey(1) & 0xFF
    #     # press 'q' to quit the window
    #     if key == ord('q'):
    #         break
    
    window_name = "Click each room label to select"
    room_titles = []
    #Mouse event
    def selectLabels(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            #print('click')
            for rect in selection_squares:
                if x > rect[0] and x < rect[2] and y > rect[1] and y < rect[3] and rect not in room_titles:
                    room_titles.append(rect)
    #Open window with possible labels in bounding boxes
    cv2.namedWindow(window_name, WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL)
    while True:
        cv2.imshow(window_name, img)
        cv2.setMouseCallback(window_name, selectLabels)
        #When c is pressed, continue
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break
    #Paint selected squares on image
    labels_selected_image = drawRect(img, room_titles)
    #Get centers of each label
    centers = getRectCenters(room_titles)
    filled_rooms_image = fillRooms(lines_image, centers)
    totalArea = getArea(filled_rooms_image)
    totalInches = convertPixeltoInch(totalArea, dpi)
    print(totalInches)
    print(totalInches*(1/float(scale_factor))**2)
    #print(totalArea)
    #key = cv2.waitKey(1) & 0xFF
    while True:
        cv2.imshow(window_name, labels_selected_image)
        cv2.imshow('test', filled_rooms_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    

if __name__ == "__main__":
    main()