from pdf2image import convert_from_path
import cv2
import numpy as np
import math
import sys
from shapely.geometry import LineString
#Base way to manipulate pdf
# pdfFileObj = open('blueprints/2bedhouse1.pdf', 'rb')

# pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# pdfFileObj.close()

#Way to make pdf an image
#Requires poppler
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

def main():
    if(len(sys.argv) < 2):
        return
    img = None
    if(sys.argv[1] == '-c'):
        #print('here')
        img =  cv2.imread('./cropped_images/' + sys.argv[2] + '.jpg', -1)
    else:
        img =  cv2.imread('./blueprint_images/' + sys.argv[1] + '.jpg', -1)
    # images = convert_from_path('blueprints/'+sys.argv[1]+'.pdf')
    # for i in range(len(images)):
    #     # Save pages as images in the pdf
    #     images[i].save('page'+ str(i) +'.jpg', 'JPEG')
    #     current_page = images[i]
    
    width = img.shape[1]/2
    height = img.shape[0]/2
    #img = cv2.resize(img,(720,1024))
    img = image_resize(img, height=1024)
    print(img.shape)
    #img = img[30:-30, 30:-30]
    cv2.imshow("image",img)
    cv2.waitKey(0)
    
    #Color squares
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = img
    kernel = np.ones((2,2),np.uint8)
    
    d = cv2.dilate(img, kernel, iterations = 2)
    # cv2.imshow("image",d)
    # cv2.waitKey(0)
    e = cv2.erode(img, kernel, iterations = 2)  
    # cv2.imshow("image",e)
    # cv2.waitKey(0)
    edges = cv2.Canny(img, 100, 100)
    cv2.imshow("image",edges)
    cv2.waitKey(0)
    
    #ret, th = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
    blank_image = create_blank(img.shape[1], img.shape[0], (255,255,255))
    lines = cv2.HoughLinesP(edges, 1, np.pi/360, 115, maxLineGap=30, minLineLength=10)
    #lines_drawn = []
    horizontal_lines = []
    vertical_lines = []
    endpoints = []
    intersections = []
    extra_length = 5

    for line in lines:
        #print(line)
        x1,y1,x2,y2 = line[0]
        if(y1 == y2):
            horizontal_lines.append([x1-extra_length, y1, x2+extra_length, y2])
            endpoints.append([x1,y1])
            endpoints.append([x2,y2])
        if(x1 == x2):
            vertical_lines.append([x1, y1+extra_length, x2, y2-extra_length])
            endpoints.append([x1,y1])
            endpoints.append([x2,y2])
    #try check for 2 intersection

    for line in horizontal_lines:
        x1,y1,x2,y2 = line
        intersects = False
        for v_line in vertical_lines:
            x3,y3,x4,y4 = v_line
            if(x1 < x3 and x2 > x3 and y1 <= y3 and y1 >= y4):
                intersects = True
                # line1 = LineString((x1,y1), (x2,y2))
                # line2 = LineString((x3,y3), (x4, y4))

                # intersections.append(line1.intersection(line2))
        if(intersects):
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,0,0),3) 
    for line in vertical_lines:
        x1,y1,x2,y2 = line
        intersects = False
        for h_line in horizontal_lines:
            x3,y3,x4,y4 = h_line
            if(y1 > y3 and y2 < y3 and x1 >= x3 and x1 <= x4):
                intersects = True
                # line1 = LineString((x1,y1), (x2,y2))
                # line2 = LineString((x3,y3), (x4, y4))

                # intersections.append(line1.intersection(line2))
        if(intersects):
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,0,0),3) 
    lines_image = blank_image       
    print(intersections)
    endpoint_image = create_blank(img.shape[1], img.shape[0], (255,255,255))
    for endpoint in endpoints:
        x, y = endpoint
        cv2.circle(endpoint_image, (x,y), radius=2, color=(0, 0, 0), thickness=-1)
    cv2.imshow("image", blank_image)
    cv2.waitKey(0)
    cv2.imshow("image", endpoint_image)
    cv2.waitKey(0)

    gray = cv2.cvtColor(lines_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3)) # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    edges_hull = cv2.Canny(gray, 100, 100)

    convex_hull_image = create_blank(img.shape[1], img.shape[0], (255,255,255))
    contours, hierarchy = cv2.findContours(edges_hull, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    #sort countours by perimeter
    contours = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)
    limit = 0
    for contour in contours:
        # if(limit > 1):
        #     break
        convexHull = cv2.convexHull(contour)
        cv2.drawContours(convex_hull_image, [convexHull], -1, (255, 0, 0), 2)
        limit += 1

    # Display the final convex hull image
    cv2.imshow('ConvexHull', edges_hull)
    cv2.waitKey(0)
    cv2.imshow('ConvexHull', convex_hull_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    lines_image = blank_image

    # #Identify and draw squares
    # gray = cv2.cvtColor(blank_image,cv2.COLOR_BGR2GRAY)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)
    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     cv2.drawContours(thresh, [c], -1, (255,255,255), -1)
    # cv2.imshow("image",thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # #Morph open
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
    # cv2.imshow("image",opening)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # input_image= thresh
    # im_flood_fill = input_image.copy()
    # h, w = input_image.shape[:2]
    # mask = np.zeros((h + 2, w + 2), np.uint8)
    # im_flood_fill = im_flood_fill.astype("uint8")
    # cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    # im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    # img_out = input_image | im_flood_fill_inv
    # cv2.imshow("image",img_out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()