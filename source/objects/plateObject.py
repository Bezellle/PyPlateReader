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
