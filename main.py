from textwrap import fill
import cv2
from cv2 import WINDOW_AUTOSIZE
from cv2 import WINDOW_GUI_NORMAL
import numpy as np
import easygui
import sys
from helpers import *
from core_functions import *

def callback(x):
    pass

def showRoomResult(src, lines_image, colors):
    room_titles = []
    window_name = "Rooms Selection"
    #Mouse event
    def selectLabels(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if [x,y] not in room_titles:
                room_titles.append([x,y])
    cv2.imshow(window_name, src)

    while True:
        cv2.setMouseCallback(window_name, selectLabels)
        #Paint selected squares on image
        filled_rooms_image = fillRooms(lines_image, room_titles, colors)
        #Get number of red pixels
        totalArea = 0
        #Breakdown color by room
        for color in colors:
            totalArea += getArea(filled_rooms_image, color[0], color[1], color[2])

        overlay = mergeImages(src, filled_rooms_image, 0.3)
        cv2.imshow(window_name, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            room_titles=[]
            cv2.imshow(window_name, src)

    return totalArea

def main():
    uni_img = easygui.fileopenbox()
    
    img = cv2.imread(uni_img, 1)
    if(img is None):
        print('No Image Selected, Exiting...')
        return
    img = image_resize(img, height=1024)
    dpi = getDPI()

    edges, gray = getEdges(img, 100, 100)
    
    lines_image, endpoints, intersections = findLines(edges, maxLineGap=30, minLineLength=10, extraLength=2, thickness=1, lineThreshold=180)
    intersections_image = paintPoints(edges, intersections)
    contour_image = getContours(lines_image)

    if(len(sys.argv) > 1 and sys.argv[1] == '-d'):
        window_name = "Takeoff Software: "
        cv2.namedWindow(window_name, WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL)
        cv2.createTrackbar("Image", window_name, 0, 3, callback)
        while True:
            # get Trackbar position
            pos = cv2.getTrackbarPos("Image", window_name)
            if pos == 0:
                cv2.imshow(window_name, img)
            elif pos == 1:
                cv2.imshow(window_name, gray)
            elif pos == 2:
                cv2.imshow(window_name, edges)
            elif pos == 3:
                cv2.imshow(window_name, lines_image)
            key = cv2.waitKey(1) & 0xFF
            # press 'q' to quit the window
            if key == ord('q'):
                cv2.destroyWindow(window_name)
                break
            
    easygui.msgbox("Click each room label to select, then hit 'q' to continue, or 'r' to reset", 'Select Rooms', 'Continue')
    #Room area colors, only as many options as in this list
    colors = [[0,0,255], [255,0,0], [0,255,0], [125, 125, 0], [0, 125, 125]]
    finalArea = showRoomResult(img, lines_image, colors)
    scale_factor = "1"
    scale_factor = easygui.enterbox("Enter Scale Factor", "Scale Factor", "1")
    totalInches = convertPixeltoSquareFeet(finalArea, dpi, scale_factor)
    easygui.msgbox('The total number of pixels in the selected rooms is: '+ str(finalArea) + '\nThe approximation of square feet selected is: ' + str(totalInches), 'Final Results', 'Quit')

if __name__ == "__main__":
    main()