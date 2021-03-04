import cv2
import numpy as np
from source.metadataExt import MetaData


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

    def __init__(self, box, objectID, plate_string=''):
        self.ObjectID = objectID
        
        self.X = box[0]
        self.Y = box[1]
        self.W = box[2] - box[0]
        self.H = box[3] - box[1]

        self.Centr = (int((box[0] + box[2])/2.0), int((box[1] + box[3])/2))

        self.PlatesDict = {}
        if plate_string != '':
            self.PlatesDict[plate_string] = 1
        #TODO: Store More Data (frame number, pos history, GPS?)

    def updateDict(self, plate_string):
        #Check length of detected string (polish license plate should be shorter than 8 chars)
        #and if it wasn't detected yet add to Detected Plate dict
        if len(plate_string) > 8:
            return
        if len(plate_string) < 5:
            return
        
        d = self.PlatesDict.get(plate_string, False)
        if not d:
            self.PlatesDict[plate_string] = 1
        else:
            self.PlatesDict[plate_string] += 1

    def getPlateNumber(self):
        #check if there are any plate numbers detections
        if len(self.PlatesDict) > 0:
            plateNumber = max(self.PlatesDict, key=self.PlatesDict.get)
        else:
            return None, 0

        #filter out detection with less than 3 repetitions
        if self.PlatesDict[plateNumber] < 3:
            return "unknown", 0
        return plateNumber, self.PlatesDict[plateNumber]

    def newPosition(self, box):
        self.X = box[0]
        self.Y = box[1]
        self.W = box[2] - box[0]
        self.H = box[3] - box[1]

        self.Centr = (int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2))

###################################################################################


class ObjectsSet(object):
    MIN_PLATE_LENGTH = 4
    MAX_PLATE_LENGTH = 8

    def __init__(self):
        self.ObjectsDict = {}
        self.ResultDict = {}

        self.MetaData = MetaData()

    def updateObjectDict(self, det_results, frame_number=0):
        #detection results has structure: [(id, box, plate_string), (...)]
        for single_det in det_results:
            #first check if plate_string is valid according to polish law
            if len(single_det[2]) < self.MIN_PLATE_LENGTH:
                break
            if len(single_det[2]) > self.MAX_PLATE_LENGTH:
                break

            #check if objectID is in the set. if is update object with new detection, otherwise create new object
            check = self.ObjectsDict.get(single_det[0], False)
            if not check:
                self.ObjectsDict[single_det[0]] = PlateObject(single_det[1], single_det[0], single_det[2])
            else:
                self.ObjectsDict[single_det[0]].updateDict(single_det[2])
                self.ObjectsDict[single_det[0]].newPosition(single_det[1])

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
    def loadMetaData(self, file_path, total_frames):
        self.MetaData.loadBin(file_path)
        self.MetaData.TotalFrames = total_frames


