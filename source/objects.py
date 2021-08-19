import cv2
import numpy as np
from source.metadataExt import MetaData
from pathlib import Path
from math import fabs


class Contour:
    """Class that represent contour of the letter - store all info about it (Position on img, size, area, etc.)"""

    def __init__(self, con=np.ones(0), img_height=1):
        # If no values are submitted with declaration __init__ creates empty object with default parameters.
        # empty contour object is created at the beginning of plate detection

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
        # This method compere if one contour leis in another
        left_corner = False
        right_corner = False

        if self.X <= Cont.X <= self.X + self.Width and \
                self.Y <= Cont.Y <= self.Y + self.Height:
            left_corner = True
        
        if self.X <= Cont.X + Cont.Width <= self.X + self.Width and\
                self.Y <= Cont.Y + Cont.Height <= self.Y + self.Height:
            right_corner = True

        if left_corner and right_corner:
            return True
        else:
            return False

    def CheckPossibility(self, imgH):
        # Check if all criteria for letter possibility are met
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


class PlateObject:
    """Store all details about detected plate object"""
    ImgCenter = 1200

    def __init__(self, box, objectID, plate_string=''):
        self.ObjectID = objectID
        
        self.X = box[0]
        self.Y = box[1]
        self.W = box[2] - box[0]
        self.H = box[3] - box[1]

        self.Centr = (int((box[0] + box[2])/2.0), int((box[1] + box[3])/2))

        self.Localization = [None]*2   # Camera localization history for first and last appearance of object
        self.PlatesDict = {}           # All detected and valid plate numbers
        self.ImgPositions = []         # All positions of object on the img (Centr)
        if plate_string != '':
            self.PlatesDict[plate_string] = 1
        # TODO: Store More Data (frame number, pos history, GPS?)

    def updateDict(self, plate_string):
        # Check if Plate number was detected, if not add new one

        d = self.PlatesDict.get(plate_string, False)
        if not d:
            self.PlatesDict[plate_string] = 1
        else:
            self.PlatesDict[plate_string] += 1

    def updateLocation(self, location):
        # Store GPS data of camera position to calculate object position
        # For now collect first and last appearance of object and calculate mean value
        # TODO: add location correctness
        if self.Localization[0] is None:
            self.Localization[0] = location
        else:
            self.Localization[1] = location

    def getLocation(self):
        ret_location = []

        if self.Localization[0] is None or self.Localization[1] is None:
            return [0, 0]

        for i in range(len(self.Localization[0])):
            new_pos = (self.Localization[0][i]+self.Localization[1][i])/2
            ret_location.append(new_pos)

        return ret_location

    def getPlateNumber(self):
        # Check if there are any plate numbers detections
        if len(self.PlatesDict) > 0:
            plate_number = max(self.PlatesDict, key=self.PlatesDict.get)
        else:
            return None, 0

        # filter out detection with less than 3 repetitions
        if self.PlatesDict[plate_number] < 3:
            return "unknown", 0
        return plate_number, self.PlatesDict[plate_number]

    def newPosition(self, box):
        # Update position of Object on the img

        self.X = box[0]
        self.Y = box[1]
        self.W = box[2] - box[0]
        self.H = box[3] - box[1]

        self.Centr = (int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2))
        self.ImgPositions.append(self.Centr)


class ObjectsSet(MetaData):
    MIN_PLATE_LENGTH = 4
    MAX_PLATE_LENGTH = 8

    def __init__(self, files_path, frame_size=(2704, 2624)):
        super().__init__()
        self.ObjectsDict = {}
        self.ResultDict = {}
        self.FilesPaths = []
        if type(files_path) == str:
            self.FilesPaths.append(Path(files_path))
        else:
            for path in files_path:
                self.FilesPaths.append(Path(path))

        PlateObject.ImgCenter = frame_size[0]/2

        self.loadMetaData(files_path)

    def __del__(self):
        super().__del__()

    def updateObjectDict(self, det_results, frame_number=0):
        if len(det_results) == 0:
            return

        if len(self.GPS) > 0:
            camera_location, direction = self.getCameraLocation(frame_number)

        # Detection results has structure: [(id, box, plate_string), (...)]
        for single_det in det_results:
            # Check if plate_string is valid according to polish law
            if len(single_det[2]) < self.MIN_PLATE_LENGTH:
                continue
            if len(single_det[2]) > self.MAX_PLATE_LENGTH:
                continue

            # Check if objectID is in the set. if it is update object with new detection, otherwise create new object
            check = self.ObjectsDict.get(single_det[0], False)
            if not check:
                self.ObjectsDict[single_det[0]] = PlateObject(single_det[1], single_det[0], single_det[2])
            else:
                self.ObjectsDict[single_det[0]].updateDict(single_det[2])
                self.ObjectsDict[single_det[0]].newPosition(single_det[1])
                if len(self.GPS) > 0:
                    self.ObjectsDict[single_det[0]].updateLocation(camera_location)

    def setResultDict(self):
        # Create dict of most detected plate numbers for all detected objects. This is the final result of detection
        for obj in self.ObjectsDict.values():
            # key - plate_string val - number of detection
            key, val = obj.getPlateNumber()
            location = obj.getLocation()
            if key is not None:
                ret = self.ResultDict.get(key, False)
                if not ret:
                    self.ResultDict[key] = [val, location]
                else:
                    self.ResultDict[key][0] += val

    def saveResults(self):
        if len(self.ResultDict) == 0:
            self.setResultDict()

        if len(self.ResultDict) == 0:
            print("No Plates recorded")
            return
        else:
            msg = "\n\nSummary:\n\tTotal Objects detected: {}\n\tTotal Plates detected: {}"
            file_path = str(self.FilesPaths[0].with_suffix('.txt'))

            print(msg.format(len(self.ObjectsDict), len(self.ResultDict)))

            with open(file_path, "w") as file:
                file.write("Plate_nuber nuber_of_detection x y\n")
                for plate, results in self.ResultDict.items():
                    if results[0] > 0:
                        file.write(str(plate) + " " + str(results[0]) + " " + str(results[1][0]) + " " + str(results[1][1]) + "\n")

                file.write(msg.format(len(self.ObjectsDict), len(self.ResultDict)))

    def loadMetaData(self, file_paths):
        # Only video from front camera store gps metadata
        for txt_file in file_paths:
            file = Path(txt_file)
            if file.name[2:4] == 'FR':
                self.loadGPSData(file)

    def setFramesNumber(self, total_frames):
        # update
        if total_frames > 0:
            self.TotalFrames = total_frames
        else:
            raise Exception('Wrong number of frames! Frame numbers have to be greater than 0')



