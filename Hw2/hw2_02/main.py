from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import mainWindow_ui as ui
import numpy as np
import cv2
import keyboard



class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)         

        self.pushButton.clicked.connect(self.corner)
        self.pushButton_2.clicked.connect(self.intrinsic)
        self.pushButton_3.clicked.connect(self.extrinsic)
        self.pushButton_4.clicked.connect(self.distortion)
        self.pushButton_5.clicked.connect(self.result)

    def corner(self):
        cv2.namedWindow("Chessboard", cv2.WINDOW_NORMAL)
        imageList = []

        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane

        # Load the image
        for i in range(1,16):
            image = cv2.imread("Dataset_OpenCvDl_Hw2/Q2_Image/" + str(i) + ".bmp")
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
            imageList.append(image)   
            


            retval, corners = cv2.findChessboardCorners(image, (8,11), flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if retval:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                chessboardImage = cv2.drawChessboardCorners(image, (8,11), corners2, retval)            
            else:
                print("No Checkerboard Found")

        ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)      
        self.ret = ret  
        self.instrincMatrix = cameraMatrix        
        self.distortion = dist
        self.rotationalvector = rvecs
        self.translationalvectors = tvecs    

        i = 0
        while True:
            i = (i + 1) % 15
            cv2.imshow("Chessboard", imageList[i])
            cv2.waitKey(500)

            if keyboard.is_pressed("q"):
                cv2.destroyAllWindows()
                break
            if keyboard.is_pressed("p"):
                cv2.waitKey(0)
            #if cv2.getWindowProperty("Chessboard", 0) == -1:
            #    break


        #video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 2, (512,  512))
        #for i in range(0,15):
        #    video.write(imageList[i])


    def intrinsic(self):
        print(self.instrincMatrix)
    
    def extrinsic(self):
        index = self.spinBox.value()
        #print(index)
        index = index - 1

        R = np.zeros((3, 3))
        cv2.Rodrigues(self.rotationalvector[index],R)

        extrinsic = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                if j != 3:
                    extrinsic[i,j] = R[i,j]
                else:
                    extrinsic[i,j] = self.translationalvectors[index][i]
        print(extrinsic)

    def distortion(self):
        print(self.distortion)

    def result(self):
        ret = self.ret
        mtx = self.instrincMatrix      
        dist = self.distortion
        rvecs = self.rotationalvector
        tvecs = self.translationalvectors
        undistoredList = []

        for i in range(15):
            #take a new image
            img = cv2.imread("Dataset_OpenCvDl_Hw2/Q2_Image/" + str(i+1) + ".bmp")
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            h,  w = img.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

            # undistort
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        
            # crop the image
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]
            #cv2.cv2.imshow('calibresult',dst)
            undistoredList.append(dst)

        distoredList = []
        for i in range(1,16):
            image = cv2.imread("Dataset_OpenCvDl_Hw2/Q2_Image/" + str(i) + ".bmp")
            image = cv2.resize(image, (492, 492), interpolation=cv2.INTER_AREA)
            distoredList.append(image)   

        i = 0
        while True:
            i = (i + 1) % 15
            numpy_horizontal = np.hstack((distoredList[i], undistoredList[i]))
            cv2.imshow("Distored vs Undistored", numpy_horizontal)
            cv2.waitKey(500)

            if keyboard.is_pressed("q"):
                cv2.destroyAllWindows()
                break
            if keyboard.is_pressed("p"):
                cv2.waitKey(0)


    


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())