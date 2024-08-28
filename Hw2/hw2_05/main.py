from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys

import matplotlib.image as mpimg
import mainWindow_ui as ui
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from random_eraser import get_random_eraser

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)         

        self.pushButton.clicked.connect(self.modleStructure)
        self.pushButton_2.clicked.connect(self.tensorBoard)
        self.pushButton_3.clicked.connect(self.test)
        self.pushButton_4.clicked.connect(self.dataAugmantation)

    def modleStructure(self):
        IMAGE_SIZE = (224, 224)
        model = ResNet50(include_top=False, weights='imagenet', input_tensor=None,input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
        print(model.summary())

    def tensorBoard(self):
        #plt.imshow(mpimg.imread('plot5-2.JPG'))
        #plt.show()
        img = cv2.imread('plot5-2.JPG')
        cv2.imshow('prob 5-2', img)

    def test(self):        
        model = ResNet50(include_top=True, weights='imagenet')
        text = self.plainTextEdit.toPlainText()
        print(text)
        #       text
        #cats   0~1249
        #dogs   1250~2499

        index = int(text)
        img_path = ""
        if index < 1250:
            img_path = "sample/test/cats/"
            index = index + 11250
        else:
            img_path = "sample/test/dogs/"
            index = index + 10000
            
        img_path = img_path + str(index) + ".jpg"

        # 載入訓練好的模型
        net = load_model('model-resnet50-final.h5')
        cls_list = ['cats', 'dogs']

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        pred = net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        if pred[0] > pred[1]:
            print("predict: cat")
            title = "class: cat"
        else:
            print("predict: dog")
            title = "class: dog"

        print(img_path)
        plt.title(title)
        plt.imshow(img)
        plt.show()

    def dataAugmantation(self):
        #plt.imshow(mpimg.imread('plot5-4.JPG'))
        #plt.show()
        img = cv2.imread('plot5-4.JPG')
        cv2.imshow('prob 5-4', img)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())