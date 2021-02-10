from calibration import Calibration 
from detection import Detection
import cv2
import numpy as np
import glob
import time

#test_path=glob.glob('D:\\praca\\GoProFusion\\15.01.2020\\*.jpg')
test_path=glob.glob('D:\\praca\\PyPlateReader\\DataSet\\train\\*.jpg')


cal=Calibration()
#cal.calibrateVideo()
cal.loadCameraParam(cal.VideoParamPath)

det=Detection()      #enable setYolo()
det.setYoloTensor()

frameCount=0
emptyFrames=0

video = True
images = False

#start_time = time.time()
#det.findLetters(None)
#end_time = time.time() - start_time

#print("Detection time: ", str(end_time))

if video:
    cap=cv2.VideoCapture("GPFR1846.MP4")
    start_frame_number = 50
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    while(cap.isOpened()):
        ret, undist=cap.read()
        frameCount += 1

        if frameCount%5 != 0:
            continue

        if ret == False:
            print("Empty Frames: ", str(frameCount))
            break
        else:
            #frame = undist
            frame = cal.undistort(undist)
            result, boxes=det.detectYoloTensor(frame)
            det.findLetters(frame,boxes)

            frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
            cv2.imshow("frame", frame)
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
    
    cap.release()
elif images == True and video == False:
    for pic in test_path:
        img=cv2.imread(pic,cv2.IMREAD_UNCHANGED)
        img=cal.undistort(img)

        #result whole img with drawbox on plates, plates - cropped img of plate
        result, plates=det.detectYoloTensor(img)
 
        det.findLetters(plates)
        frameCount += 1
        
        #if frameCount==50:
        #    break
else:
    print("Wrong Test Params")
    
#det.saveResults()


if video==True:
    cap.release()


cv2.destroyAllWindows()
