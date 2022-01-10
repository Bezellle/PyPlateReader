from source.metadataExt import MetaData
from pathlib import Path
from source.objects.plateObject import PlateObject


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

    def loadMetaData(self, file_path):
        # Only video from front camera store gps metadata
            file = Path(file_path)
            if file.name[2:4] == 'FR':
                self.loadGPSData(file)

    def setFramesNumber(self, total_frames):
        # update
        if total_frames > 0:
            self.TotalFrames = total_frames
        else:
            raise Exception('Wrong number of frames! Frame numbers have to be greater than 0')
