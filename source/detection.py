import cv2
import numpy as np
import tensorflow as tf
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from easydict import EasyDict as edict
import time
import glob
import copy

from source.tracker import Tracker
from source.objects import Contour, PlateObject
from source.customOCR import CustomOCR


class Detection:
    """description of class"""
    def __init__(self, display=False):
        self.classes = ["plate"]
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        print(physical_devices)
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.CharNumber = 0
        self.PlateNum = 0
        self.EmptyFrame = 0
        self.FrameNumber = 0

        self.FLAG = edict() 
        self.FLAG.size = 416
        self.FLAG.tiny = False
        self.FLAG.model = "yolov4"
        #self.FLAG.weights = './model/custom-416'
        self.FLAG.weights = './model/modelYolov4'
        self.FLAG.framework = "tf"
        self.FLAG.display = display

        self.OCR = CustomOCR()
        self.Track = Tracker(maxMissing=5, maxDistance=225)

    @staticmethod
    def preprocess(img):
        MORPH_KERNEL = np.ones((3, 3), np.uint8)      #kernel for morph operations
        #img=self.rotate(src,-15)

        #TODO: Update preprocessing for bigger plate image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #print("Char image for recognition should be in grayscale! \n Size has been changed!")
        #img=self.maximizeContrast(img)

        img = cv2.GaussianBlur(img, (5, 5), 0)

        if img.shape[1] < 600:
            img_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 4)
            img_thresh = cv2.dilate(img_thresh, MORPH_KERNEL, iterations=1)
            img_thresh = cv2.erode(img_thresh, MORPH_KERNEL, iterations=2)
        else:
            img_thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            img_thresh = cv2.bitwise_not(img_thresh)
            img_thresh = cv2.dilate(img_thresh, MORPH_KERNEL, iterations=1)

        img_thresh = cv2.dilate(img_thresh, MORPH_KERNEL, iterations=1)
        #img_thresh = cv2.erode(img_thresh, MORPH_KERNEL, iterations=2)
        #img_thresh = cv2.dilate(imgThresh, MORPH_KERNEL, iterations=1)
        return img_thresh

    def setYoloTensor(self):
        #detect plate with tensorflow framework and GPU device

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(self.FLAG)

        start_time = time.time()
        print("Loading started")

        self.saved_model_loaded = tf.saved_model.load(self.FLAG.weights, tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']

        end_time = time.time()-start_time
        print("Model loaded in: ", end_time)

    def detectYoloTensor(self, frame):
        if frame is None:
            print("No frame loaded")
        else:
            frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frameRGB)

        self.FrameNumber += 1

        image_data = cv2.resize(frameRGB,(self.FLAG.size, self.FLAG.size))
        image_data = image_data/255
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)        #all boxes found by net
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        #combined boxes based on IOU
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.25
            )

        #convert tensors to numpy arrays
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]  
        
        image = utils.draw_bbox(frame, pred_bbox)
        image = Image.fromarray(image.astype(np.uint8))
        result = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        rects = self.cutBox(frame, pred_bbox, False)
        
        #result is a frame with boxes
        #rects are coordinates of detected boxes
        return result, rects

    def cutBox(self, frame, predbox, save=False):
        ## Crop detected plate from orginal image ##

        frame_w, frame_h=frame.shape[:2]
        boxes, scores, classes, num_boxes=predbox
        border = 5

        rects = []
        for i in range(num_boxes[0]):
            coor = boxes[0][i]
            coor[0] = int(coor[0])
            coor[2] = int(coor[2])
            coor[1] = int(coor[1])
            coor[3] = int(coor[3])

            if coor[0] - border < 0 or coor[0] + border > frame_h:
                continue
            if coor[2] - border < 0 or coor[2] + border > frame_h:
                continue
            if coor[1] - border < 0 or coor[2] + border > frame_w:
                continue
            if coor[2] - border < 0 or coor[2] + border > frame_w:
                continue

            # Box structure: (Xstart, Ystart, Xend, Yend)
            rects.append((int(coor[1])-border, int(coor[0])-border, int(coor[3])+border, int(coor[2])+border))

        # Saving plates image
        if save:
            path = './DataSet/Plates/'
            plates = []

            for i, box in enumerate(rects):
                plate = frame[box[1]:box[3], box[0]:box[2]]
                cv2.imwrite(path + str(self.FrameNumber) + '_' + str(i) + '.jpg', plate)
        
        return rects

    def findLetters(self, img=None, boxes=None):
        # plate examples for test
        #if img == None and boxes == None: plates=glob.glob('./DataSet/Plates/*.jpg')

        error_num = 0
        #result_list is result of the method. it stores tuples of (is, box, plate_string)
        result_list = []

        ret, trackedIDs = self.Track.update(boxes)
        #ret = False when there are no objects for detection
        if not ret:
            return result_list

        for id in list(trackedIDs.keys()):
            boxId = list(trackedIDs[id])[1]
            if boxId == -1:
                continue

            #TODO: Change image exraction for boxes coord due to different method logic
            plt = img[boxes[boxId][1]: boxes[boxId][3], boxes[boxId][0]: boxes[boxId][2]]

            try:  
                plt = cv2.resize(plt, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            except:
                print("Error occured during findLetters func\nEmpty frame cannot be resized!\n ")
                continue

            #Check if image is blurry. If True - skip
            mean, blurry = self.blurryFFT(plt, size=35, thresh=-15)
            if blurry:
                if self.FLAG.display:
                    text = "Blurry ({:.4f})"
                    text = text.format(mean)
                    cv2.putText(plt, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [200, 200, 0], 2)
                else:
                    print("Blurry image detected")
                    #continue

            if self.FLAG.display:
                cv2.imshow("Source", plt)
                cv2.waitKey(1)

            # apply: threshold, gaussian blur and morph
            plt = self.preprocess(plt)

            #TODO: Change contours to blobs!
            #find contours on image: with canny edges first
            edges = cv2.Canny(plt, 100, 200)
            try:
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except:
                ret, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #sort contours (from left to right) 
            sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])    

            plate_string = ""

            # to avoid error at lieIn func on 1st contours store empty con as previous
            ctr_prev = Contour()
            for con in sorted_contours:
                img_height, img_width = plt.shape[:2]
                ctr = Contour(con, img_height)
                #minRect=cv2.minAreaRect(con)

                if self.FLAG.display:
                    #temp is object just for tests
                    temp = copy.deepcopy(plt)
                    temp = cv2.rectangle(temp, (ctr.X - 5, ctr.Y - 5), (ctr.X + ctr.Width + 5, ctr.Y + ctr.Height + 5),
                                         (125, 125, 125), 2)
                    cv2.imshow("plate", temp)
                    cv2.waitKey(1)

                #criteria for letter possibility:
                if not ctr.PossibleLetter:
                    continue

                #Check if contour is inside of previous contour
                if ctr_prev.lieIn(ctr):
                    ctr.PossibleLetter = False
                    continue        

                ctr_prev = Contour(con, img_height)
                
                #rotated rect
                #box = cv2.boxPoints(minRect)
                #box = np.intp(box)

                #Extract single char from plate image. 
                char = plt[ctr.Y - 5:ctr.Y + ctr.Height + 5, ctr.X - 5:ctr.X + ctr.Width + 5]  #RECTANGULAR
                
                if char is None:
                    continue
                
                #char=self.cropSubPix(plt,minRect)      #MIN AREA RECT
                try:
                    charStr = self.OCR.prediction(char)
                    
                    plate_string += charStr  
                    
                    if self.FLAG.display:
                        print(charStr)
                        cv2.imshow("char", char)
                        cv2.waitKey(0)
                    
                    cv2.imwrite('./DataSet/Chars/pack3/'+str(self.CharNumber)+'.jpg', char)
                    self.CharNumber += 1
                except:
                    print("Unexpected Error\n")
                    error_num += 1

                char = None

            print("Frame: ", self.FrameNumber, "Object nr: ", str(id), " - ", plate_string)
            if error_num > 0:
                print(str(error_num))

            result_list.append((id, boxes[boxId], plate_string))

        return result_list

    def rotate(self, img, angle, center=None, scale=1.0):
    
        # grab the dimensions of the image
        (h, w) = img.shape[:2]

        # if the center is None, initialize it as the center of the image
        if center is None:
            center = (w // 2, h // 2)

        # perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(img, M, (w, h))

        # return the rotated image
        return rotated

    def cropSubPix(self, img, rect):
        center, size, angle = rect[0], rect[1], rect[2]
        center, size = tuple(map(int, center)), tuple(map(int, size))

        # get row and col num in img
        height, width = img.shape[0], img.shape[1]

        # calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # rotate the original image
        img_rot = cv2.warpAffine(img, M, (width, height))

        # now rotated rectangle becomes vertical, and we crop it
        img_crop = cv2.getRectSubPix(img_rot, size, center)

        return img_crop

    @staticmethod
    def maximizeContrast(imgGrayscale):

        height, width = imgGrayscale.shape

        imgTopHat = np.zeros((height, width, 1), np.uint8)
        imgBlackHat = np.zeros((height, width, 1), np.uint8)

        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

        imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
        imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

        return imgGrayscalePlusTopHatMinusBlackHat

    def saveIMG(self, img):
        path = './DataSet/Plates/'

        try:
            if img != None:
                cv2.imwrite(path+str(self.PlateNum)+'.jpg', img)
        except:
            self.EmptyFrame += 1
            print("Empty Frame no:" + str(self.EmptyFrame))

    def blurryFFT(self, img, size = 30, thresh = 5):
        ##  Function that detects if image is blurred or not. Returns True if it is.
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        (h, w) = img.shape
        (cX, cY) = (int(w/2), int(h/2))

        # make fft transform and shift zeros values to center
        fft = np.fft.fft2(img)
        fftShift = np.fft.fftshift(fft)

        # zero-out center of fft
        fftShift[cY - size:cY + size, cX - size: cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)

        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)

        return mean, mean <= thresh
