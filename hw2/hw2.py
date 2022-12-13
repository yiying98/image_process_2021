import sys
from PyQt5.QtWidgets import *
from hw2qt import Ui_MainWindow
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import math
from PyQt5.QtGui import QImage, QPixmap, qRgb


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.ReadImageButton.clicked.connect(self.load_data)
        self.img= None
        self.imgA = None
        self.imgB =None
        self.output = None
        self.alpha = 0
        self.beta = 0
        self.imgName=''
        self.ui.GrayScaleBButton.clicked.connect(self.grayscaleB)
        self.ui.GrayscaleAButton.clicked.connect(self.grayscaleA)
        self.ui.CompareButton.clicked.connect(self.compare)
        self.ui.ConvertButton.clicked.connect(self.setbinary)
        self.ui.SpatialSlider.setTickInterval(1)
        self.ui.SpatialSlider.setSingleStep(1)
        self.ui.SpatialSlider.valueChanged[int].connect(self.spatial)
        self.ui.BrightnessSlider.setTickInterval(10)
        self.ui.BrightnessSlider.setSingleStep(10)
        self.ui.BrightnessSlider.valueChanged[int].connect(self.brightness)
        self.ui.ContrastSlider.setTickInterval(10)
        self.ui.ContrastSlider.setSingleStep(10)
        self.ui.ContrastSlider.valueChanged[int].connect(self.contrast)
        self.ui.EqualizationButton.clicked.connect(self.equal)
        


    def load_data(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self,"開啟檔案","./","(*jpg *png);;(*.jpg *.png)") 
        if fileName1 != '':
            self.imgName = fileName1.split('/')[-1]
            self.ui.NameShow.setText(self.imgName)
            self.img = cv2.imread(fileName1)
            
            height, width, channel = self.img.shape
            bytesPerline = 3 * width            
            qimage_output = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()            
            self.ui.ImageShow.setPixmap(QPixmap.fromImage(qimage_output))

    def spatial(self,value):
        self.imgB = ((self.img[:,:,0]*0.299+self.img[:,:,1]*0.587+self.img[:,:,2]*0.114)).astype(np.uint8)

        sample = 0.1*value
        self.output = cv2.resize(self.imgB, dsize=(int(self.imgB.shape[1]*sample), int(self.imgB.shape[0]*sample)),
         interpolation=cv2.INTER_LINEAR)
        self.plot_input_histogram(self.output, 'output.png')
        self.show_output(self.output, 'output.png')

    def grayscaleB(self):
        self.imgB = ((self.img[:,:,0]*0.299+self.img[:,:,1]*0.587+self.img[:,:,2]*0.114)).astype(np.uint8)
        self.plot_input_histogram(self.imgB, 'imgB.png')
        self.show_output(self.imgB, 'imgB.png')
    
    def grayscaleA(self):
        self.imgA = ((self.img[:,:,0]/3+self.img[:,:,1]/3+self.img[:,:,2]/3)).astype(np.uint8)
        self.plot_input_histogram(self.imgA, 'imgA.png')
        self.show_output(self.imgA, 'imgA.png')
    

    def compare(self):
        self.output = (self.imgA-self.imgB).astype(np.uint8)
        self.plot_input_histogram(self.output, 'output.png')
        self.show_output(self.output, 'output.png')

    def setbinary(self):
        self.imgB = ((self.img[:,:,0]*0.299+self.img[:,:,1]*0.587+self.img[:,:,2]*0.114)).astype(np.uint8)
        threshold = int(self.ui.Threshold.text())
        self.output =np.where(self.imgB>threshold,255,0).astype(np.int8)
        self.plot_input_histogram(self.output, 'output.png')
        self.show_output(self.output, 'output.png')

    def equal(self):
        self.imgB = ((self.img[:,:,0]*0.299+self.img[:,:,1]*0.587+self.img[:,:,2]*0.114)).astype(np.uint8)
        hist,bins = np.histogram(self.imgB.ravel(),256,[0,255])
        
        pdf = hist/self.imgB.size          #出現次數/總像素點 
        cdf = pdf.cumsum()          # 將每一個灰度級的概率利用cumsum()累加
        equ_value = np.around(cdf * 255).astype('uint8')        #將cdf的結果，乘以255 

        self.output = equ_value[self.imgB].astype(np.uint8)
        self.plot_input_histogram(self.output, 'output.png')
        self.show_output(self.output, 'output.png')      


    def brightness(self,value):
        self.imgB = ((self.img[:,:,0]*0.299+self.img[:,:,1]*0.587+self.img[:,:,2]*0.114)).astype(np.uint8)
        self.alpha = value
        b = self.alpha/255.0
        c = self.beta/255.0
        k = math.tan((45 + 44 * c) / 180 * math.pi)
        
        img = (self.imgB - 127.5 * (1 - b)) * k + 127.5 * (1 + b)
        self.output = np.clip(img, 0, 255).astype(np.uint8)
        self.plot_input_histogram(self.output, 'output.png')
        self.show_output(self.output, 'output.png') 




        
    def contrast(self,value):
        self.imgB = ((self.img[:,:,0]*0.299+self.img[:,:,1]*0.587+self.img[:,:,2]*0.114)).astype(np.uint8)
        self.beta = value
        b = self.alpha/255.0
        c = self.beta/255.0
        k = math.tan((45 + 44 * c) / 180 * math.pi)
        
        img = (self.imgB - 127.5 * (1 - b)) * k + 127.5 * (1 + b)
        self.output = np.clip(img, 0, 255).astype(np.uint8)
        self.plot_input_histogram(self.output, 'output.png')
        self.show_output(self.output, 'output.png') 

    

    def plot_input_histogram(self ,array ,name):
        plt.hist(array.ravel(), 256, [0, 256])
        plt.savefig(name)
        plt.close('all')


    def show_output(self,array,hist_name):
        hist = cv2.imread(hist_name)
        qimage_output = QImage(array, array.shape[1], array.shape[0],
         array.shape[1], QImage.Format_Grayscale8)
        qimage_hist =QImage(hist.data, hist.shape[1], hist.shape[0], hist.shape[1]*3,
         QImage.Format_RGB888).rgbSwapped()

        self.ui.ImageShow.setPixmap(QPixmap.fromImage(qimage_output))
        self.ui.ImageShow.setScaledContents(True)                
        self.ui.HistogramShow.setPixmap(QPixmap.fromImage(qimage_hist))
        self.ui.HistogramShow.setScaledContents(True)




if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())