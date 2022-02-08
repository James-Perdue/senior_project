from pdf2image import convert_from_path
import cv2
import numpy as np

#Base way to manipulate pdf
# pdfFileObj = open('blueprints/2bedhouse1.pdf', 'rb')

# pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# pdfFileObj.close()

#Way to make pdf an image
#Requires poppler

images = convert_from_path('blueprints/2bedhouse1.pdf')
for i in range(len(images)):
    # Save pages as images in the pdf
    images[i].save('page'+ str(i) +'.jpg', 'JPEG')
    current_page = images[i]
    img =  cv2.imread('page'+ str(i) +'.jpg',-1)
    img = cv2.resize(img,(1024,1024))
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #Recognize dotted lines
    # kernel1 = np.ones((3,5),np.uint8)
    # kernel2 = np.ones((9,9),np.uint8)
    # imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # imgBW=cv2.threshold(imgGray, 230, 255, cv2.THRESH_BINARY_INV)[1]
    # img1=cv2.erode(imgBW, kernel1, iterations=1)
    # img2=cv2.dilate(img1, kernel2, iterations=3)
    # img3 = cv2.bitwise_and(imgBW,img2)
    # img3= cv2.bitwise_not(img3)
    # img4 = cv2.bitwise_and(imgBW,imgBW,mask=img3)
    # imgLines= cv2.HoughLinesP(img4,15,np.pi/180,10, minLineLength = 50, maxLineGap = 15)

    # for i in range(len(imgLines)):
    #     for x1,y1,x2,y2 in imgLines[i]:
    #         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    # cv2.imshow('Final Image with dotted Lines detected', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #Color squares
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 180, 200)
    kernel = np.ones((2,2),np.uint8)
    
    d = cv2.dilate(edges,kernel,iterations = 2)
    e = cv2.erode(img,kernel,iterations = 2)  
    #ret, th = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,30, maxLineGap=40,minLineLength=120)
    for line in lines:
        #print(line)
        x1,y1,x2,y2 = line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()