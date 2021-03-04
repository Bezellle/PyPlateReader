from source.calibration import Calibration
from source.detection import Detection
from source.fps import FPSTracker
from source.metadataExt import MetaData
from source.objects import ObjectsSet
import cv2
import glob

img_path = glob.glob('.\\DataSet\\train\\*.jpg')
test_path = glob.glob('.\\test\\*.MP4')
fps_logger = FPSTracker("Frame_load", "Calibration", "Yolo_Detection", "Plate_reading", "Dataset_update", "Total")

cal = Calibration()
cal.loadCameraParam(cal.VideoParamPath)

det = Detection(display=False)
det.setYoloTensor()
objDataSet = ObjectsSet()

emptyFrames = 0

video = True
images = False

fps_logger["Total"].start()

if video:
    for path in test_path:
        cap = cv2.VideoCapture(path)
        meta = MetaData()
        # meta.load_data("GPFR3074.MP4")
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(total_frames)
        next_frame = total_frames - 50

        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        while(cap.isOpened()):
            fps_logger["Frame_load"].start()
            ret, undist = cap.read()
            fps_logger["Frame_load"].stop()

            #Skip empty frames end break the loop at the end of video
            if next_frame < int(total_frames) and not ret:
                emptyFrames += 1
                next_frame += 3
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
                print("Number of empty frames detected: ", str(emptyFrames))
                continue
            elif next_frame >= total_frames and not ret:
                print("Number of empty frames detected: ", str(emptyFrames))
                break
            else:
                #Start detection
                frame = undist
                fps_logger["Calibration"].start()
                #frame = cal.undistort(undist)
                frame = cal.cutout(undist)
                fps_logger["Calibration"].stop()

                fps_logger["Yolo_Detection"].start()
                result, boxes = det.detectYoloTensor(frame)
                fps_logger["Yolo_Detection"].stop()

                fps_logger["Plate_reading"].start()
                result_list = det.findLetters(frame, boxes)
                fps_logger["Plate_reading"].stop()

                fps_logger["Dataset_update"].start()
                objDataSet.updateObjectDict(result_list, next_frame)
                fps_logger["Dataset_update"].stop()

                frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
                cv2.imshow("frame", frame)
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break

                next_frame += 3
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

        cap.release()
elif images and not video:
    for pic in test_path:
        img = cv2.imread(pic, cv2.IMREAD_UNCHANGED)
        img = cal.undistort(img)

        #result whole img with drawbox on plates, plates - cropped img of plate
        result, plates = det.detectYoloTensor(img)
 
        det.findLetters(plates)

else:
    print("Wrong Test Params")

fps_logger["Total"].stop()
objDataSet.saveResults()

fps_logger.saveLog()
if video:
    cap.release()

cv2.destroyAllWindows()
