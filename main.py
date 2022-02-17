from pdf2image import convert_from_path
import cv2
import numpy as np
import math
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

def main():
    images = convert_from_path('blueprints/2bedhouse1.pdf')
    #images = convert_from_path('blueprints/2bedV2.pdf')
    #images = convert_from_path('blueprints/20x45-Model.pdf')
    for i in range(len(images)):
        # Save pages as images in the pdf
        images[i].save('page'+ str(i) +'.jpg', 'JPEG')
        current_page = images[i]
        img =  cv2.imread('page'+ str(i) +'.jpg',-1)
        #img = cv2.resize(img,(1024,1024))

        img = img[30:-30, 30:-30]
        cv2.imshow("image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #Color squares
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, 220, 350)
        kernel = np.ones((2,2),np.uint8)
        
        d = cv2.dilate(edges, kernel, iterations = 2)
        e = cv2.erode(img, kernel, iterations = 2)  
        #ret, th = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
        blank_image = create_blank(img.shape[1], img.shape[0], (255,255,255))
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,30, maxLineGap=30,minLineLength=120)
        lines_drawn = []
        for line in lines:
            #print(line)
            x1,y1,x2,y2 = line[0]
            #print(line[0])
            if(x1 == x2 or y1 == y2):
                # for line_draw in lines_drawn:
                #     #print(line_draw)
                #     x3,y3,x4,y4 = line_draw
                #     line_tolerance = 0
                #     if((abs(x1-x3) < line_tolerance and abs(x2-x4) < line_tolerance or abs(y1-y3) < line_tolerance and abs(y2-y4) < line_tolerance) and line_length(line[0]) > line_length(line_draw)):

                #         lines_drawn.remove(line_draw)
                #         lines_drawn.append([x1,y1,x2,y2])
                #print(lines_drawn)
                lines_drawn.append([x1,y1,x2,y2])

        for line in lines_drawn:
            x1,y1,x2,y2 = line
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,0,0),3)
                    
        cv2.imshow("image",blank_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        lines_image = blank_image

        #Identify and draw squares
        gray = cv2.cvtColor(blank_image,cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(thresh, [c], -1, (255,255,255), -1)
        cv2.imshow("image",thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #Morph open
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
        cv2.imshow("image",opening)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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