from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import mainWindow_ui as ui
import numpy as np
import cv2
from scipy import signal
from scipy import ndimage




class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)         

        self.pushButton.clicked.connect(self.loadImage)
        self.pushButton_2.clicked.connect(self.colorSeperation)
        self.pushButton_3.clicked.connect(self.colorTransformation)
        self.pushButton_4.clicked.connect(self.blending)
        self.pushButton_5.clicked.connect(self.gaussianBlur)
        self.pushButton_6.clicked.connect(self.bilateralFilter)
        self.pushButton_7.clicked.connect(self.medianFilter)
        self.pushButton_8.clicked.connect(self.gaussianBlur2)
        self.pushButton_9.clicked.connect(self.sobelX)
        self.pushButton_10.clicked.connect(self.sobelY)
        self.pushButton_11.clicked.connect(self.magnitude)
        self.pushButton_12.clicked.connect(self.resizeImg)
        self.pushButton_13.clicked.connect(self.translation)
        self.pushButton_14.clicked.connect(self.rotationScaling)
        self.pushButton_15.clicked.connect(self.shearing)

        self.center = None


    #problem 1
    def loadImage(self):
        fname = QFileDialog.getOpenFileName(self, "Open file")#, 'c:\\',"Image files (*.jpg *.gif)")
        #print(fname)#type: tuple

        self.img = cv2.imread(fname[0])
        h,w,c = self.img.shape #row (height) x column (width) x color (3)
        print("height: " + str(h) + " \tweight:" + str(w))
        
        #cv2.namedWindow("Load Image", cv2.WINDOW_NORMAL)

        cv2.imshow("Load Image", self.img)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #cv2.destroyWindow("Load Image")

    def colorSeperation(self):
        b,g,r = cv2.split(self.img)
        zeros = np.zeros(self.img.shape[:2],dtype="uint8")

        #cv2.namedWindow("blue", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("green", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("red", cv2.WINDOW_NORMAL)

        cv2.imshow("blue", cv2.merge([b,zeros,zeros]))
        cv2.imshow("green", cv2.merge([zeros,g,zeros]))
        cv2.imshow("red", cv2.merge([zeros,zeros,r]))

        #cv2.waitKey(0)
        #cv2.destroyWindow("blue")
        #cv2.destroyWindow("green")
        #cv2.destroyWindow("red")

    def colorTransformation(self):
        grayscale = cv2.cvtColor(self.img,  cv2.COLOR_BGR2GRAY)
        
        b,g,r = cv2.split(self.img)
        averageWeight = np.zeros(self.img.shape[:2],dtype="uint8")
        for row in range(self.img.shape[0]):
            for col in range(self.img.shape[1]):                
                average = ( int(b[row][col]) + int(g[row][col]) + int(r[row][col]))/3                               
                averageWeight[row][col] = int(average)    

        #cv2.namedWindow("grayscale", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("averageWeight", cv2.WINDOW_NORMAL)

        cv2.imshow("grayscale", grayscale)
        cv2.imshow("averageWeight", averageWeight)
        
        #cv2.waitKey(0)
        #cv2.destroyWindow("grayscale")
        #cv2.destroyWindow("averageWeight")

    def blending(self):
        #fname = QFileDialog.getOpenFileName(self, "Open file")
        #img1 = cv2.imread(fname[0])
        fname = QFileDialog.getOpenFileName(self, "Open file")
        self.img2 = cv2.imread(fname[0])

        cv2.namedWindow("blending")
        cv2.createTrackbar("blend value", "blending", 0, 255, self.update)
        cv2.setTrackbarPos("blend value", "blending", 125)

    def update(self,value):
        value = cv2.getTrackbarPos("blend value", "blending")
        value = (value + 1)/256.0
        #print(value)

        dst = np.zeros(self.img.shape[:2],dtype="uint8")
        dst=cv2.addWeighted(self.img, 1-value, self.img2, value, 0)
        
        cv2.imshow("blending", dst)

    #problem 2
    def gaussianBlur(self):
        if self.img is None:
            self.img = cv2.imread("Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_pepperSalt.jpg")

        gaussian = cv2.GaussianBlur(self.img, (5,5), 0)
        cv2.imshow("Gaussian Blurred Image", gaussian)

    def bilateralFilter(self):
        self.img = cv2.imread("Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_pepperSalt.jpg")

        bilateral = cv2.bilateralFilter(self.img, 9, 90, 90)
        cv2.imshow("Bilateral Filter", bilateral)

    def medianFilter(self):
        self.img = cv2.imread("Dataset_OpenCvDl_Hw1\Q2_Image\Lenna_whiteNoise.jpg")

        median3 = cv2.medianBlur(self.img, 3)
        median5 = cv2.medianBlur(self.img, 5)
        cv2.imshow("Median filter 3x3", median3)
        cv2.imshow("Median filter 5x5", median5)

    #problem 3
    def gaussianBlur2(self):
        if self.img is None:
            self.img = cv2.imread("Dataset_OpenCvDl_Hw1\Q3_Image\House.jpg")
        grayscale = cv2.cvtColor(self.img,  cv2.COLOR_BGR2GRAY)
        cv2.imshow("grayscale", grayscale)
        

        #3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        #print(gaussian_kernel)

        grad = signal.convolve2d(grayscale, gaussian_kernel,  boundary='symm', mode='same')

        
        (h, w) = self.img.shape[:2]
        for row in range(h):
            for col in range(w):
                if grad[row,col] > 255:
                    grad[row,col] = 255
                if grad[row,col] < 0:
                    grad[row,col] = 0

        grad = grad.astype(np.uint8)
        cv2.imshow("Gaussian Blur", grad)
        
    def convolution(self,A,B):
        #C = np.multiply(A,B)
        s = 0
        for i in range(0,3):
            for j in range(0,3):
                #s = s + C[i,j]
                s = s + A[i,j] * B[2-i, 2-j]

        if s < 0:
            s = 0 - s   
        if s > 255:
            s = 255        
          
        return s 

    def sobelX(self):
        grayscale = cv2.cvtColor(self.img,  cv2.COLOR_BGR2GRAY)
        result = np.copy(grayscale)
        Gx = np.float32([[-1,0,1], [-2,0,2], [-1,0,1]])
        (h, w) = self.img.shape[:2]
        
        for row in range(0,h-2):
            for col in range(0,w-2):
                tmp = np.array([[grayscale[row,col],  grayscale[row,col+1],   grayscale[row,col+2]], 
                                [grayscale[row+1,col],grayscale[row+1,col+1], grayscale[row+1,col+2]],
                                [grayscale[row+2,col],grayscale[row+2,col+1], grayscale[row+2,col+2]]])     
                result[row+1,col+1] = self.convolution(tmp, Gx)        
        
        result = result.astype(np.uint8)
        cv2.imshow("sobel X", result)



        
    def sobelY(self):
        grayscale = cv2.cvtColor(self.img,  cv2.COLOR_BGR2GRAY)
        result = np.copy(grayscale)
        Gy = np.float32([[-1,-2,-1], [0,0,0], [1,2,1]])
        (h, w) = self.img.shape[:2]

        for row in range(0,h-2):
            for col in range(0,w-2):
                tmp = np.array([[grayscale[row,col],  grayscale[row,col+1],   grayscale[row,col+2]], 
                                [grayscale[row+1,col],grayscale[row+1,col+1], grayscale[row+1,col+2]],
                                [grayscale[row+2,col],grayscale[row+2,col+1], grayscale[row+2,col+2]]])                
                result[row+1,col+1] = self.convolution(tmp, Gy)                
        
        result = result.astype(np.uint8)
        cv2.imshow("sobel Y", result)

        
        
    def magnitude(self):
        grayscale = cv2.cvtColor(self.img,  cv2.COLOR_BGR2GRAY)
        result = np.copy(grayscale)

        Gx = np.float32([[-1,0,1], [-2,0,2], [-1,0,1]])
        Gy = np.float32([[-1,-2,-1], [0,0,0], [1,2,1]])
        (h, w) = self.img.shape[:2]

        for row in range(0,h-2):
            for col in range(0,w-2):
                tmp = np.array([[grayscale[row,col],  grayscale[row,col+1],   grayscale[row,col+2]], 
                                [grayscale[row+1,col],grayscale[row+1,col+1], grayscale[row+1,col+2]],
                                [grayscale[row+2,col],grayscale[row+2,col+1], grayscale[row+2,col+2]]])         
                gx = self.convolution(tmp, Gx)
                gy = self.convolution(tmp, Gy)
                g = np.sqrt(gx * gx + gy * gy)
                if g > 255:
                    g = 255
                if g < 0:
                    g = 0
                result[row+1,col+1] = g
        
        result = result.astype(np.uint8)
        cv2.imshow("sobel magnitude", result)
        

    #problem 4
    def resizeImg(self):
        if self.img is None:
            self.img = cv2.imread("Dataset_OpenCvDl_Hw1\Q4_Image\SQUARE-01.png")
        
        
        resize_image = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)

        #cv2.namedWindow("Resize Image", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Resize Image", resize_image)

        self.img = resize_image

        self.center = [128, 128]

    def translation(self):
        M = np.float32([[1, 0, 0],[0, 1, 60]])

        (h, w) = self.img.shape[:2]
        #shifted_image = cv2.warpAffine(self.img, M, (w, h+60))
        shifted_image = cv2.warpAffine(self.img, M, (400, 300))
       
        cv2.imshow("Translation Image", shifted_image)
        #cv2.resizeWindow("Translation Image", 400, 300)

        self.img = shifted_image

        if self.center is None:
            self.center = [w/2, h/2]
        
        self.center[1] = self.center[1] + 60
        
    def rotationScaling(self):
        angle=10
        scale=0.5

        (h, w) = self.img.shape[:2]                 

        M = cv2.getRotationMatrix2D(tuple(self.center) , angle, scale)
        rotated_image = cv2.warpAffine(self.img, M, (400, 300))        

        #cv2.namedWindow("Rotated Image", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Rotated Image", rotated_image)
        #cv2.resizeWindow("Rotated Image", 400, 300)

        self.img = rotated_image
        
    def shearing(self):    

        src = np.float32([[50,50],[200,50],[50,200]])
        dst = np.float32([[10,100],[200,50],[100,250]])
        M = cv2.getAffineTransform(src,dst)
        #print (M)
        
        (h, w) = self.img.shape[:2]
        aff2_image = cv2.warpAffine(self.img, M, (w, h))

        #cv2.namedWindow("Shearing Image", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Shearing Image", aff2_image)

        self.img = aff2_image
        
        


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())