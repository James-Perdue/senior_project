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
    # scale_factor = "test"
    # scale_factor = easygui.enterbox("Enter Scale Factor", "Scale Factor", "")
    img = cv2.imread(uni_img, 1)

    img = image_resize(img, height=1024)
    dpi = getDPI()
    print(dpi)

    edges, gray = getEdges(img, 100, 100)
    
    lines_image, endpoints, intersections = findLines(edges, maxLineGap=30, minLineLength=10, extraLength=2, thickness=1, lineThreshold=180)
    blue_gone = findNonBlue(edges, img, 30, 10, 5, 1)
    endpoint_image = paintPoints(edges, endpoints)
    intersections_image = paintPoints(edges, intersections)

    contour_image = getContours(lines_image)

    window_name = "Takeoff Software: "
    cv2.namedWindow(window_name, WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL)
    cv2.createTrackbar("Image", window_name, 0, 5, callback)
    while True:
        # get Trackbar position
        pos = cv2.getTrackbarPos("Image", window_name)
        if pos == 0:
            cv2.imshow(window_name, img)
        elif pos == 1:
            cv2.imshow(window_name, edges)
        elif pos == 2:
            cv2.imshow(window_name, lines_image)
        elif pos == 3:
            cv2.imshow(window_name, blue_gone)
        elif pos == 4:
            cv2.imshow(window_name, endpoint_image)
        elif pos == 5:
            cv2.imshow(window_name, contour_image)
        key = cv2.waitKey(1) & 0xFF
        # press 'q' to quit the window
        if key == ord('q'):
            break
    
    window_name = "Select Rooms then hit C to continue"
    room_titles = []
    #Mouse event
    def selectLabels(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if [x,y] not in room_titles:
                room_titles.append([x,y])
    #Open window with possible labels in bounding boxes
    easygui.msgbox('Click each room label to select', 'Select Rooms', 'Continue')
    cv2.namedWindow(window_name, WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL)
    while True:
        cv2.imshow(window_name, img)
        cv2.setMouseCallback(window_name, selectLabels)
        #When c is pressed, continue
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break
    #Paint selected squares on image
    filled_rooms_image = fillRooms(lines_image, room_titles)
    totalArea = getArea(filled_rooms_image)
    totalInches = convertPixeltoInch(totalArea, dpi)
    print(totalArea)
    print(totalInches)
    #key = cv2.waitKey(1) & 0xFF
    overlay = mergeImages(img, filled_rooms_image, 0.3)
    while True:
        #cv2.imshow(window_name, labels_selected_image)
        cv2.imshow('test', overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    

if __name__ == "__main__":
    main()