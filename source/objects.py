import cv2
import numpy as np
from source.metadataExt import MetaData
from math import fabs


class Contour(object):
    """description of class"""

    def __init__(self, con=np.ones(0), img_height=1):
        #TODO description of default param
        if con.shape[0] == 0:
            x, y, w, h = [1, 1, 1, 1]
            self.Area = 0
        else:
            x, y, w, h = cv2.boundingRect(con)
            self.Area = cv2.contourArea(con)

        self.X = x
        self.Y = y
        self.Width = w
        self.Height = h
        self.Center = (x+w/2, y+h/2)
        self.Ratio = h/float(w)

        self.RectArea = self.Width * self.Height
        self.PossibleLetter = self.CheckPossibility(img_height)

    def lieIn(self, Cont):
        left_corner = False
        right_corner = False

        if self.X <= Cont.X <= self.X + self.Width and self.Y <= Cont.Y <= self.Y + self.Height:
            left_corner = True
        
        if self.X <= Cont.X + Cont.Width <= self.X + self.Width and\
                self.Y <= Cont.Y + Cont.Height <= self.Y + self.Height:
            right_corner = True

        if left_corner and right_corner:
            return True
        else:
            return False

    def CheckPossibility(self, imgH):
        #Check if all criteria for letter possibility are met
        if self.Ratio < 1.3:
           return False
        elif self.Ratio > 3.9:
           return False
        elif imgH/self.Height > 4:
           return False
        elif self.RectArea < 500:
           return False
        elif self.X == 0:
           return False
        elif self.Y == 0:
           return False
        else:
            return True

##############################################################################


class PlateObject(object):
    """Store all details about detected plate object"""
    ImgCenter = 1200

    def __init__(self, box, objectID, plate_string=''):
        self.ObjectID = objectID
        
        self.X = box[0]
        self.Y = box[1]
        self.W = box[2] - box[0]
        self.H = box[3] - box[1]

        self.Centr = (int((box[0] + box[2])/2.0), int((box[1] + box[3])/2))

        self.Localization = []       # All camera localization history
        self.PlatesDict = {}         # All detected and valid plate numbers
        self.ImgPositions = []       # All positions of object on the img (Centr)
        if plate_string != '':
            self.PlatesDict[plate_string] = 1
        #TODO: Store More Data (frame number, pos history, GPS?)

    def updateDict(self, plate_string):
        #Check if Plate number was detected, if not add new one

        d = self.PlatesDict.get(plate_string, False)
        if not d:
            self.PlatesDict[plate_string] = 1
        else:
            self.PlatesDict[plate_string] += 1

    def updateLocation(self, location):
        #store GPS data of camera position to calculate object position next. Limit to 6 positions

        if len(self.Localization) <= 6:
            self.Localization.append(location)

    def getPlateNumber(self):
        #check if there are any plate numbers detections
        if len(self.PlatesDict) > 0:
            plate_number = max(self.PlatesDict, key=self.PlatesDict.get)
        else:
            return None, 0

        #filter out detection with less than 3 repetitions
        if self.PlatesDict[plate_number] < 3:
            return "unknown", 0
        return plate_number, self.PlatesDict[plate_number]

    def newPosition(self, box):
        self.X = box[0]
        self.Y = box[1]
        self.W = box[2] - box[0]
        self.H = box[3] - box[1]

        self.Centr = (int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2))
        self.ImgPositions.append(self.Centr)

###################################################################################


class ObjectsSet(object):
    MIN_PLATE_LENGTH = 4
    MAX_PLATE_LENGTH = 8

    def __init__(self, frame_size=(2704, 2624)):
        self.ObjectsDict = {}
        self.ResultDict = {}
        PlateObject.ImgCenter = frame_size[0]/2

        self.MetaData = MetaData()

    def updateObjectDict(self, det_results, frame_number=0):
        if len(det_results) == 0:
            return

        camera_location, direction = self.MetaData.getCameraLocation(frame_number)

        #detection results has structure: [(id, box, plate_string), (...)]
        for single_det in det_results:
            #Check if plate_string is valid according to polish law
            if len(single_det[2]) < self.MIN_PLATE_LENGTH:
                break
            if len(single_det[2]) > self.MAX_PLATE_LENGTH:
                break

            #check if objectID is in the set. if it is update object with new detection, otherwise create new object
            check = self.ObjectsDict.get(single_det[0], False)
            if not check:
                self.ObjectsDict[single_det[0]] = PlateObject(single_det[1], single_det[0], single_det[2])
            else:
                self.ObjectsDict[single_det[0]].updateDict(single_det[2])
                self.ObjectsDict[single_det[0]].newPosition(single_det[1])
                self.ObjectsDict[single_det[0]].updateLocation(camera_location)

    def setResultDict(self):
        #Create dict of most detected plate numbers for all detected objects. This is the final result of detection
        for id in range(len(self.ObjectsDict)):
            #key - plate_string val - number of detection
            key, val = self.ObjectsDict[id].getPlateNumber()
            if key is not None:
                ret = self.ResultDict.get(key, False)
                if not ret:
                    self.ResultDict[key] = val
                else:
                    self.ResultDict[key] += val

    def saveResults(self):
        if len(self.ResultDict) == 0:
            self.setResultDict()

        if len(self.ResultDict) == 0:
            print("No Plates recorded")
            return
        else:
            msg = "\n\nSummary:\n\tTotal Objects detected: {}\n\tTotal Plates detected: {}"

            print(msg.format(len(self.ObjectsDict), len(self.ResultDict)))

            with open("./log/testResult.txt", "w") as file:
                for plate, qty in self.ResultDict.items():
                    if qty > 0:
                        file.write(str(plate) + " " + str(qty) + "\n")

                file.write(msg.format(len(self.ObjectsDict), len(self.ResultDict)))

    # TODO: Add GPS
    def loadMetaData(self, file_paths):
        #Only video from front camera store gps metadata
        for file in file_paths:
            txt_split = file.split("\\")
            if txt_split[-1][2:4] == 'FR':
                self.MetaData.loadGPSData(file)

    def setFramesNumber(self, total_frames):
        #update
        if total_frames > 0:
            self.MetaData.TotalFrames = total_frames
        else:
            raise Exception('Wrong number of frames! Frame numbers have to be greater than 0')



