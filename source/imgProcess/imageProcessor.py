import cv2
import numpy as np


class ImageProcessor:
    MORPH_KERNEL = np.ones((3, 3), np.uint8)

    def preprocess(self, img):
        # preprocess method perform all image preprocessing - change to grayscale, thresholding, gaussianblur etc.
        #img=self.rotate(src,-15)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #print("Char image for recognition should be in grayscale! \n Size has been changed!")
        #img=self.maximizeContrast(img)

        img = cv2.GaussianBlur(img, (5, 5), 0)

        #apply OTSU for bigger plates
        if img.shape[1] < 600:
            img_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 4)
            img_thresh = cv2.dilate(img_thresh, self.MORPH_KERNEL, iterations=1)
            img_thresh = cv2.erode(img_thresh, self.MORPH_KERNEL, iterations=2)
        else:
            img_thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            img_thresh = cv2.bitwise_not(img_thresh)
            img_thresh = cv2.dilate(img_thresh, self.MORPH_KERNEL, iterations=1)

        img_thresh = cv2.dilate(img_thresh, self.MORPH_KERNEL, iterations=1)
        #img_thresh = cv2.erode(img_thresh, MORPH_KERNEL, iterations=2)
        #img_thresh = cv2.dilate(imgThresh, MORPH_KERNEL, iterations=1)
        return img_thresh
