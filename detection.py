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

from tracker import Tracker
from objects import Contour, PlateObject
from customOCR import CustomOCR


class Detection:
    """description of class"""
    def __init__(self):
        self.classes=["plate"]
        physical_devices=tf.config.experimental.list_physical_devices('GPU')
        print(physical_devices)
        if len(physical_devices)>0:
            tf.config.experimental.set_memory_growth(physical_devices[0],True)

        self.DetectedPlates = {}      #dict with uniq detected plates and number of detection
        self.CharNumber = 0
        self.PlateNum = 0
        self.EmptyFrame = 0
        self.PlatesDataset = {}

        self.FLAG = edict() 
        self.FLAG.size = 416
        self.FLAG.tiny = False
        self.FLAG.model = "yolov4"
        #self.FLAG.weights = './model/custom-416'
        self.FLAG.weights = './model/modelYolov4'
        self.FLAG.framework = "tf"
        self.FLAG.display = True
        self.Numbers = {str(i) for i in range(10)}

        self.OCR=CustomOCR()
        self.Track = Tracker(maxMissing = 5, maxDistance = 100)

    ###########image preprocessing: active treshold and erode(if specified)########
    def preprocess(self, img):
        MORPH_KERNEL=np.ones((3,3),np.uint8)      #kernel for morph operations
        #img=self.rotate(src,-15)

        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print("Char image for recognition should be in grayscale! \n Size has been changed!")
        #img=self.maximizeContrast(img)

        img=cv2.GaussianBlur(img,(5,5),0)
        
        imgTresh=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,29,2)
        #imgTresh=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        imgTresh=cv2.bitwise_not(imgTresh)
        
        imgTresh=cv2.dilate(imgTresh,MORPH_KERNEL,iterations=1)
        imgTresh=cv2.erode(imgTresh,MORPH_KERNEL,iterations=2)
        return imgTresh


#######################################################

    def setYoloTensor(self):
        #detect plate with tensorflow framework and GPU device

        config=ConfigProto()
        config.gpu_options.allow_growth = True
        session=InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(self.FLAG)

        start_time=time.time()
        print("Loading started")
        if(self.FLAG.framework=="tflite"):
           self.interpreter = tf.lite.Interpreter(model_path=self.FLAG.weights)
           self.interpreter.allocate_tensors()
           self.input_details = self.interpreter.get_input_details()
           self.output_details = self.interpreter.get_output_details()
           print(self.input_details)
           print(self.output_details)
        else:
            self.saved_model_loaded = tf.saved_model.load(self.FLAG.weights, tags=[tag_constants.SERVING])
            self.infer = self.saved_model_loaded.signatures['serving_default']

        end_time=time.time()-start_time
        print("Model loaded in: ",end_time)
      

#######################################################

    def detectYoloTensor(self, frame):
        if frame is None:
            print("No frame loaded")
        else:
            frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image=Image.fromarray(frameRGB)

        image_data=cv2.resize(frameRGB,(self.FLAG.size, self.FLAG.size))
        image_data=image_data/255
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        if(self.FLAG.framework=="tflite"):
            self.interpreter.set_tensor(input_details[0]['index'], image_data)
            self.interpreter.invoke()
            pred = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([self.FLAG.size, self.FLAG.size]))
        else:
            batch_data=tf.constant(image_data)
            pred_bbox=self.infer(batch_data)        #all boxes found by net
            for key, value in pred_bbox.items():
                boxes=value[:, :, 0:4]
                pred_conf=value[:, :, 4:]


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
        
        #result is a frame with boxes on detectons 
        #rects are coordinates of detected boxes
        return result, rects

    #######################################################

    def cutBox(self, frame, predbox, save=False):
        ## Crop detected plate from orginal image ##

        frame_w, frame_h=frame.shape[:2]
        boxes, scores, classes, num_boxes=predbox

        rects = []
        #plates = []
        for i in range(num_boxes[0]):
            coor=boxes[0][i]
            coor[0]=int(coor[0])
            coor[2]=int(coor[2])
            coor[1]=int(coor[1])
            coor[3]=int(coor[3])

            if coor[0] - 15 < 0 or coor[0] + 15 > frame_h: continue
            if coor[2] - 15 < 0 or coor[2] + 15 > frame_h: continue
            if coor[1] - 15 < 0 or coor[2] + 15 > frame_w: continue
            if coor[2] - 15 < 0 or coor[2] + 15 > frame_w: continue

            #Box structure: [Xstart, Ystart, Xend, Yend]
            rects.append([int(coor[1])-15, int(coor[0])-15,int(coor[3])+15,int(coor[2])+15])
            #cropped=frame[int(coor[0])-15:int(coor[2])+15,int(coor[1])-15:int(coor[3])+15]
            #plates.append(cropped)

        ## Saving plates image
        if save == True:
            path='./DataSet/Plates/'

            for pic in range(len(plates)):
                cv2.imwrite(path+str(pic)+'.jpg',plates[pic])
        
        return rects     

###########################################################################


    def findLetters(self, img = None, boxes = None):

        ## plate examples for test
        #if img == None and boxes == None: plates=glob.glob('./DataSet/Plates/*.jpg')

        error_num=0

        ret, trackedIDs = self.Track.update(boxes)
        if not ret: return

        for id in list(trackedIDs.keys()):
            if not self.PlatesDataset.get(id, False):
                boxId = list(trackedIDs[id])[1]
                self.PlatesDataset[id] = PlateObject(boxes[boxId], id)

            plt = img[self.PlatesDataset[id].Y : self.PlatesDataset[id].Y + self.PlatesDataset[id].H,
                      self.PlatesDataset[id].X : self.PlatesDataset[id].X + self.PlatesDataset[id].W]

            try:  
                plt = cv2.resize(plt,None,fx=3.0,fy=3.0,interpolation = cv2.INTER_CUBIC)
            except:
                print("Error occured during findLetters func\nEmpty frame cannot be resized!\n ")
                continue

            mean, blurry = self.blurryFFT(plt, size = 35, thresh = -15)
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
            
            plt=self.preprocess(plt)   #apply: threshold, gaussian blur and morph(if true)

            #find contours on image: with canny edges first
            edges = cv2.Canny(plt,100,200)
            try:
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except:
                ret, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
               

            #sort contours (from left to right) 
            sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])    

            plate_string=""

            CtrPrev=Contour(sorted_contours[0])     #to avoid error at lieIn func on 1st contours
            for con in sorted_contours:
                img_height, img_width= plt.shape[:2]
                Ctr = Contour(con, img_height)
                #minRect=cv2.minAreaRect(con)

                if self.FLAG.display:
                    #temp is object just for tests
                    temp = copy.deepcopy(plt)
                    temp = cv2.rectangle(temp,(Ctr.X-5,Ctr.Y-5),(Ctr.X+Ctr.Width+5,Ctr.Y+Ctr.Height+5),(125,125,125),2)
                    cv2.imshow("plate",temp)
                    cv2.waitKey(1)

                #criteria for letter posibility:             
                if Ctr.PossibleLetter == False: continue

                #Check if contour is inside of previous contour
                if CtrPrev.lieIn(Ctr): 
                    Ctr.PossibleLetter = False
                    continue        

                CtrPrev = Contour(con, img_height)
                
                #rotated rect
                #box=cv2.boxPoints(minRect)
                #box=np.intp(box)

                #Extract single char from plate image. 
                char=plt[Ctr.Y - 5:Ctr.Y + Ctr.Height + 5, Ctr.X - 5:Ctr.X + Ctr.Width + 5]  #RECTANGULAR
                
                if char is None: continue
                
                #char=self.cropSubPix(plt,minRect)      #MIN AREA RECT
                try:
                    charStr = self.OCR.prediction(char)
                    
                    plate_string += charStr  
                    
                    if self.FLAG.display:
                        print(charStr)
                        cv2.imshow("char",char)
                        cv2.waitKey(0)
                    
                    #cv2.imwrite('./DataSet/Chars/pack2/'+str(self.CharNumber)+'.jpg', char)
                    #self.CharNumber += 1
                except:
                    print("Unexpected Error\n")
                    error_num += 1

                char=None
                
            
            print(plate_string)
            print(str(error_num))
            #if plate_string != '': self.set_DetectedPlate(plate_string)
            if plate_string != '': self.PlatesDataset[id].updateDict(plate_string)


    def rotate(self,img,angle,center=None,scale=1.0):
    
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

    def maximizeContrast(self, imgGrayscale):

        height, width = imgGrayscale.shape

        imgTopHat = np.zeros((height, width, 1), np.uint8)
        imgBlackHat = np.zeros((height, width, 1), np.uint8)

        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

        imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
        imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

        return imgGrayscalePlusTopHatMinusBlackHat

    def set_DetectedPlate(self):
        for id in len(self.PlatesDataset):
            key, val = self.PlateDataser[id].getPlateNumber()
            self.DetectedPlates[key] = val
    


    def saveResults(self):
        if len(self.DetectedPlates) == 0:
            self.set_DetecedPlate()
        if len(self.DetectedPlates) == 0:
            print("No Plates recorded")
            return

        with open("testResult.txt","w") as file:
            for plate, qty in self.DetectedPlates.items():
                if qty>0:
                    file.write(str(plate) + " " + str(qty) + "\n")

    def saveIMG(self, img):
        path='./DataSet/Plates/'

        try:
            if img != None:
                cv2.imwrite(path+str(self.PlateNum)+'.jpg', img)
        except:
            self.EmptyFrame += 1
            print("Empty Frame no:" + str(self.EmptyFrame ))

    def blurryFFT(self, img, size = 30, thresh = 5):
        ##   Function that detects if image is blurred or not. Returns True if it is. 
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        (h, w) = img.shape
        (cX, cY) = (int(w/2), int(h/2))

        # make fft transform and shift zeros values to center
        fft = np.fft.fft2(img)
        fftShift = np.fft.fftshift(fft)

        # zero-out center of fft
        fftShift[cY - size:cY + size, cX - size: cX +size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)

        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)

        return mean, mean <= thresh
        
            