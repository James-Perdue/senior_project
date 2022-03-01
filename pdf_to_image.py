from pdf2image import convert_from_path
import cv2
import numpy as np
import math
import sys
import os

#single page currently
def main():
    if(len(sys.argv) != 2):
        for filename in os.listdir('./blueprints'):
            images = convert_from_path('blueprints/'+filename)
            
            images[0].save('./blueprint_images/' + filename[:-4] + '.jpg', 'JPEG')
        return
    images = convert_from_path('blueprints/'+sys.argv[1]+'.pdf')
    for i in range(len(images)):
        # Save pages as images in the pdf
        images[i].save(sys.argv[1] +'p' + str(i) + '.jpg', 'JPEG')

if __name__ == "__main__":
    main()