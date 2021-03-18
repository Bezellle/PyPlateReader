import cv2
import numpy as np
#assert cv2.__version__[0] <= '3'

import glob

class Calibration(object):
    CutOutBorderX = 200
    CutOutBorderY = 600

    def __init__(self, method='cutout'):
        self.K = np.zeros((3, 3))
        self.D = np.zeros((4, 1))
        self.flag = 0     #flag indicates if calibration param are loaded flag=1 -> param loaded

        self.VideoParamPath = "VideoCaliParam.txt"
        if method == 'cutout':
            self.Method = method
        elif method == 'undistort':
            self.Method = method
        else:
            assert "Wrong calibration method has been chosen. Available oprions: 'cutout' or 'undistort'"


#########End of constructor###################################


    def calibrate(self):
        CHECKERBOARD = (6, 9)
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        _img_shape = None
        images = glob.glob('calibrationIMG/*.jpg')

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for fname in images:
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)
        
        N_OK = len(objpoints)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(objpoints,
                imgpoints,
                gray.shape[::-1],
                self.K,
                self.D,
                rvecs,
                tvecs, 
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        
        
        print(_img_shape[::-1])
        self.Dim = _img_shape[::-1]
        print(self.Dim)
        
        self.saveParam()    #save calibration parameters to file
        self.flag=1

#print("Found " + str(N_OK) + " valid images for calibration")
#print("DIM=" + str(_img_shape[::-1]))
#print("K=np.array(" + str(K.tolist()) + ")")
#print("D=np.array(" + str(D.tolist()) + ")")

#########End of calibrate func###################################

    def calibrateVideo(self):
        CHECKERBOARD = (6, 9)
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        _img_shape = None
        
        videoPath = 'calibrationIMG/calibration.mp4'
        cap = cv2.VideoCapture(videoPath)
        
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        frameC = 0
        while cap.isOpened():
            ret, img = cap.read()
            frameC += 1
        
            if frameC % 10 != 0:
                continue

            if not ret:
                break

            if _img_shape is None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
                imgpoints.append(corners)
        
        N_OK = len(objpoints)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(objpoints,
                imgpoints,
                gray.shape[::-1],
                self.K,
                self.D,
                rvecs,
                tvecs, 
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        
        
        print(_img_shape[::-1])
        self.Dim = _img_shape[::-1]
        print(self.Dim)
        
        self.saveParam(self.VideoParamPath)    #save calibration parameters to file
        self.flag = 1

        cap.release()
#print("Found " + str(N_OK) + " valid images for calibration")
#print("DIM=" + str(_img_shape[::-1]))
#print("K=np.array(" + str(K.tolist()) + ")")
#print("D=np.array(" + str(D.tolist()) + ")")



    def saveParam(self, path="testCali.txt"):
        with open(path,"w") as file:
            file.write(str(self.Dim[0]) + " " + str(self.Dim[1]) + "\n")
            for item in self.D:
                for val in item:
                    file.write(str(val) + " ")
            file.write("\n")
            for item in self.K:
                for val in item:
                    file.write(str(val) + " ")
                file.write("\n")
#########End of saveParam func###################################

    def loadCameraParam(self, path="testCali.txt" ):          #Load Resoltultion (Dim) Camera Matrix (K) and Dceoff
        with open(path, "r") as file:
             dim = [int(x) for x in next(file).split()]
             allLines = file.readlines()

        self.Dim = tuple(dim)
        d = allLines[0]
        d = d[:-3]
        self.D = np.array([float(x) for x in d.replace("\n","").split(" ")])       #stored as a numpy array for opencv

        k = allLines[1:]
        k = [i[:-3] for i in k]
        self.K = np.array([[float(x) for x in line.replace("\n","").split(" ")] for line in k])
        self.flag = 1

#########End of loadCameraParam###################################
    def undistort(self, img, dim2=None, dim3=None):
        #img = cv2.imread(img_path)
        if self.flag == 1:
            dim1 = img.shape[:2][::-1]

            assert dim1[0]/dim1[1] == self.Dim[0]/self.Dim[1], "Different image Ratio"

            if not dim2:
                dim2 = dim1
            if not dim3:
                dim3 = dim1

            scaled_K = self.K*dim1[0]/self.Dim[0]
          
            #new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, self.D, dim2, np.eye(3), balance=0.0)
            #print(new_K,"\n\n",scaled_K)

            map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), scaled_K,dim1, cv2.CV_16SC2)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            #map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.Dim, cv2.CV_16SC2)
            #undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            print("No camera param loaded")
        #cv2.imshow("undistorted", undistorted_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return undistorted_img

    def cutout(self, img):
        img_h, img_w = img.shape[:2]

        cut_out = img[self.CutOutBorderY:img_h - self.CutOutBorderY, self.CutOutBorderX:img_w - self.CutOutBorderX]
        return cut_out

    def calibrateIMG(self, img):
        if self.Method == 'undistort':
            result = self.undistort(img)
        else:
            result = self.cutout(img)

        return  result

    def getImgSize(self):
        if self.Method == 'undistort':
            return self.Dim
        elif self.Method == 'cutout':
            return (self.Dim[1]-2*self.CutOutBorderY, self.Dim[0]-2*self.CutOutBorderX)
