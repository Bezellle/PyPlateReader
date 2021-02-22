from calibration import Calibration 
from detection import Detection
from fps import FPSTracker
import cv2
import glob
import time

test_path = glob.glob('.\\DataSet\\train\\*.jpg')
fps_logger = FPSTracker("Frame_load", "Calibration", "Yolo_Detection", "Plate_reading", "Total")

cal = Calibration()
cal.loadCameraParam(cal.VideoParamPath)

det = Detection(display=False)
det.setYoloTensor()

frameCount = 0
emptyFrames = 0

video = True
images = False

fps_logger["Total"].start()
# TODO: frame controll / Timestamp
if video:
    cap = cv2.VideoCapture("GPFR1846.MP4")
    start_frame_number = 50
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    while(cap.isOpened()):
        fps_logger["Frame_load"].start()
        ret, undist = cap.read()
        fps_logger["Frame_load"].stop()

        frameCount += 1

        if frameCount % 4 != 0:
            continue

        # if frameCount == 75:
        #     break

        if not ret:
            print("Empty Frames: ", str(frameCount))
            break
        else:
            frame = undist
            fps_logger["Calibration"].start()
            # frame = cal.undistort(undist)
            frame = cal.cutout(undist)
            fps_logger["Calibration"].stop()

            fps_logger["Yolo_Detection"].start()
            result, boxes = det.detectYoloTensor(frame)
            fps_logger["Yolo_Detection"].stop()

            fps_logger["Plate_reading"].start()
            det.findLetters(frame, boxes)
            fps_logger["Plate_reading"].stop()

            # frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
            # cv2.imshow("frame", frame)
            # if cv2.waitKey(2) & 0xFF == ord('q'):
            #     break
    
    cap.release()
elif images and not video:
    for pic in test_path:
        img=cv2.imread(pic, cv2.IMREAD_UNCHANGED)
        img=cal.undistort(img)

        #result whole img with drawbox on plates, plates - cropped img of plate
        result, plates = det.detectYoloTensor(img)
 
        det.findLetters(plates)
        frameCount += 12
        
        #if frameCount==50:
        #    break
else:
    print("Wrong Test Params")

fps_logger["Total"].stop()
det.saveResults()

fps_logger.saveLog()
if video:
    cap.release()

cv2.destroyAllWindows()
