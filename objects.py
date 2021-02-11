import cv2

class Contour(object):
    """description of class"""

    def __init__(self, con, imgHeight=1):
        x, y, w, h=cv2.boundingRect(con)
        
        self.X=x
        self.Y=y
        self.Width=w
        self.Height=h
        self.Center = (x+w/2, y+h/2)

        self.Ratio=h/float(w)
        self.Area=cv2.contourArea(con)
        self.RectArea=self.Width * self.Height
        self.PossibleLetter = self.CheckPossibility(imgHeight)


    def lieIn(self, Cont):
        left_corner = False
        right_corner = False

        if self.X <= Cont.X <= self.X + self.Width and self.Y <= Cont.Y <= self.Y + self.Height:
            left_corner = True
        
        if self.X <= Cont.X + Cont.Width <= self.X + self.Width and self.Y <= Cont.Y + Cont.Height <= self.Y + self.Height:
            right_corner = True

        if left_corner and right_corner:
            return True
        else:
            return False

    def CheckPossibility(self, imgH):
        if self.Ratio < 1.2:
           return False
        elif self.Ratio > 3.5:
           return False
        elif imgH/self.Height > 4:
           return False
        elif self.RectArea < 300:
           return False
        elif self.X == 0:
           return False
        elif self.Y == 0:
           return False
        else:
            return True

class PlateObject(object):
    """Store all details about detected plate object"""

    def __init__(self, box, objectID):
        self.ObjectID = objectID
        
        self.X = box[0]
        self.Y = box[1]
        self.W = box[2] - box[0]
        self.H = box[3] - box[1]

        self.Centr = (int((box[0] + box[2])/2.0), int((box[1] + box[3])/2))

        self.PlatesDict = {}

    def updateDict(self, PlateString):
        #Check length of detected string (polish license plate should be shorter than 8 chars)
        #and if it wasn't detected yet add to Detected Plate dict
        if len(PlateString) > 8: return
        if len(PlateString) < 5: return
        
        d = self.PlatesDict.get(PlateString, False)
        if not d: self.PlatesDict[PlateString] = 1
        else: self.PlatesDict[PlateString] += 1

    def getPlateNumber(self):
        plateNumber = max(self.PlatesDict, key=self.PlatesDict.get)
        return plateNumber, self.PlatesDict[plateNumber]

    def newPosition(self, box):
        self.X = box[0]
        self.Y = box[1]
        self.W = box[2] - box[0]
        self.H = box[3] - box[1]

        self.Centr = (int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2))
