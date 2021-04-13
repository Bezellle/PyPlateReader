from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class Tracker(object):
    """Euclidean tracker class to label each plate objects"""

    def __init__(self, maxMissing=5, maxDistance=50):
        # ID is individual number for each detected and tracked object
        # Objects have structure [id, array(centerX, centerY), col],
        # where col is object number in input boxes
        self.NextObjectID = 0
        self.Objects = OrderedDict()
        self.Missing = OrderedDict()

        # max number of missing object that are out of scope
        self.MaxMissing = maxMissing

        self.MaxDistance = maxDistance

    def add(self, center, col):
        #add new object for tracking
        self.Objects[self.NextObjectID] = [center, col]
        self.Missing[self.NextObjectID] = 0
        self.NextObjectID += 1

    def delOutOfScope(self, objectId):
        #delete objects that left frame
        del self.Objects[objectId]
        del self.Missing[objectId]

    def update(self, rects):
        #check if there are no objects detected if True add to missing or 
        #go out of tracker scope if number of missed frames is higher than thresh

        if len(rects) == 0:
            for objectID in list(self.Objects.keys()):
                self.Missing[objectID] += 1
                if self.Missing[objectID] > self.MaxMissing:
                    self.delOutOfScope(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return False, self.Objects

        #Initialize an array for Centeroids 
        objectIds = list(self.Objects.keys())
        objectCentroids = np.array([i[0] for i in list(self.Objects.values())])
        inputCentroids = np.zeros((len(rects), 2), dtype="int")


        #calculate centroids of given rects
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        #if there are no objects in tracker scope add all rects
        if len(self.Objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.add(inputCentroids[i], i)
        else:
            #Calculate distance beetween given rects centers and tracking objects 
            D = dist.cdist(objectCentroids, inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.MaxDistance:
                    continue

                ID = objectIds[row]
                self.Objects[ID] = [inputCentroids[col], col]
                self.Missing[ID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            #If there is more rows than cols that means there is less points on new frame
            #Additional points have to go out of tracking scope
            
            # loop over the unused row indexes to check if something is not assigned
            # -1 means point is missing in current frame but still in scope
            if len(unusedRows) > 0:
                for row in unusedRows:
                    ID = objectIds[row]
                    self.Missing[ID] += 1
                    if self.Missing[ID] > self.MaxMissing:
                        self.delOutOfScope(ID)
                    else:
                        self.Objects[ID][1] = -1

            if len(unusedCols) > 0:
                for col in unusedCols:
                    self.add(inputCentroids[col], col)

        return True, self.Objects
