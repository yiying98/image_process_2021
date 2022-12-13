import sys
from PyQt5.QtWidgets import *
from hw3qt import Ui_MainWindow
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import math
from scipy import signal
from PyQt5.QtGui import QImage, QPixmap, qRgb
import numpy as np



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.OpenPicture.clicked.connect(self.load_data)
        self.ui.Box.clicked.connect(self.Box)
        self.ui.Gaussian.clicked.connect(self.Gaussian)
        self.ui.Laplacian.clicked.connect(self.Laplacian)
        self.ui.MarrHildreth.clicked.connect(self.LOG)
        self.ui.Sobel.clicked.connect(self.Sobel)
        self.ui.MinFilter.clicked.connect(self.MinFilter)
        self.ui.MaxFilter.clicked.connect(self.MaxFilter)
        self.ui.MedianFilter.clicked.connect(self.MedianFilter)
        self.imgName=None
        self.img= None
        self.output = None

        


    def load_data(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self,"開啟檔案","./","(*JPG *png *jpeg);;(*.jpg *.JPG *.JPEG)") 
        if fileName1 != '':
            self.imgName = fileName1.split('/')[-1]
            self.ui.NameShow.setText(self.imgName)
            img = cv2.imread(fileName1)
            self.img = ((img[:,:,0]/3+img[:,:,1]/3+img[:,:,2]/3)).astype(np.uint8)
            qimage_output = QImage(self.img, self.img.shape[1], self.img.shape[0],self.img.shape[1], QImage.Format_Grayscale8)
            self.ui.ImageShow.setPixmap(QPixmap.fromImage(qimage_output))
            self.ui.ImageShow.setScaledContents(True)                

    def Box(self):
        BoxSize = 3 if self.ui.KernelSize.text()=="" else int(self.ui.KernelSize.text())
        kernel = np.ones((BoxSize,BoxSize))
        kernel = kernel/kernel.sum()
        self.output = signal.convolve2d(self.img, kernel, boundary='symm', mode='same').astype(np.uint8)
        self.show_output()

    def Gaussian(self):
        
        GaussianSize = 3 if self.ui.KernelSize.text()=="" else int(self.ui.KernelSize.text())
        kernel = self.GaussianKernel(GaussianSize)
        self.output = signal.convolve2d(self.img, kernel, boundary='symm', mode='same').astype(np.uint8)
        self.show_output()

    def GaussianKernel(self,GaussianSize):
        sig=math.sqrt(0.5)
        ax = np.linspace(-(GaussianSize - 1) / 2., (GaussianSize - 1) / 2., GaussianSize)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
        kernel = kernel/kernel.sum()
        return kernel


    def Laplacian(self):
        LaplacianSize = 3 if self.ui.KernelSize.text()=="" else int(self.ui.KernelSize.text())
        kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        self.output = signal.convolve2d(self.img, kernel, boundary='symm', mode='same').astype(np.uint8)
        self.output = cv2.convertScaleAbs(self.output, alpha=255/self.output.max())
        cv2.imwrite('Laplacian.png',self.output)
        #self.output = cv2.Laplacian(self.img, cv2.CV_16S, ksize=LaplacianSize)
        #self.output = cv2.convertScaleAbs(self.output)
        self.show_output()

    def LOG(self):
        '''Gkernel = self.GaussianKernel(3)
        Lkernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
        GaussianOut = signal.convolve2d(self.img, Gkernel, boundary='symm', mode='same').astype(np.uint8)
        self.output = signal.convolve2d(GaussianOut, Lkernel, boundary='symm', mode='same').astype(np.uint8)
        self.output = cv2.convertScaleAbs(self.output,alpha=255/self.output.max())
        self.show_output()'''
        sig=math.sqrt(0.5)
        GaussianSize = 3 if self.ui.KernelSize.text()=="" else int(self.ui.KernelSize.text())
        ax = np.linspace(-(GaussianSize - 1) / 2., (GaussianSize - 1) / 2., GaussianSize)
        one = np.ones((len(ax),len(ax)))
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.multiply(one-(np.square(xx) + np.square(yy))/(2*np.square(sig)),np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig)))
        self.output = signal.convolve2d(self.img, kernel, boundary='symm', mode='same').astype(np.uint8)
        self.output = cv2.convertScaleAbs(self.output,alpha=255/self.output.max())
        cv2.imwrite('LOG.png',self.output)
        self.show_output()
    
    def Sobel(self):
        kernel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        x = signal.convolve2d(self.img, kernel_x, boundary='symm', mode='same').astype(np.uint8)
        x = cv2.convertScaleAbs(x,alpha=255/x.max())
        y = signal.convolve2d(self.img, kernel_y, boundary='symm', mode='same').astype(np.uint8)
        y = cv2.convertScaleAbs(y,alpha=255/y.max())
        self.output = np.add(y/2,x/2).astype(np.uint8)
        self.show_output()
        '''x = cv2.Sobel(self.img, cv2.CV_16S, 1, 0, ksize=3) 
        y = cv2.Sobel(self.img, cv2.CV_16S, 0, 1, ksize=3)
        absX = cv2.convertScaleAbs(x) 
        absY = cv2.convertScaleAbs(y)
        self.output = cv2.addWeighted(absX, 0.5, absY,0.5,0)
        self.show_output()'''

    def MinFilter(self):
        KernelSize = 3 if self.ui.KernelSize.text()=="" else int(self.ui.KernelSize.text())
        PadSize= KernelSize//2
        PadImg = np.pad(self.img,((PadSize,PadSize),(PadSize,PadSize)),'edge')
        self.output = np.zeros((self.img.shape[0],self.img.shape[1]))


        for i in range(PadSize,PadImg.shape[0]-PadSize):
            for j in range(PadSize,PadImg.shape[1]-PadSize):                
                self.output[i-PadSize,j-PadSize] = 255-PadImg[i-PadSize:i-PadSize+KernelSize,j-PadSize:j-PadSize+KernelSize].min()
        
        self.output = self.output.astype(np.uint8)
        cv2.imwrite('Min.png',self.output)
        self.show_output()


    def MaxFilter(self):
        KernelSize = 3 if self.ui.KernelSize.text()=="" else int(self.ui.KernelSize.text())
        PadSize= KernelSize//2
        PadImg = np.pad(self.img,((PadSize,PadSize),(PadSize,PadSize)),'edge')
        #print(PadImg)
        self.output = np.zeros((self.img.shape[0],self.img.shape[1]))


        for i in range(PadSize,PadImg.shape[0]-PadSize):
            for j in range(PadSize,PadImg.shape[1]-PadSize):                
                self.output[i-PadSize,j-PadSize] = np.max(PadImg[i-PadSize:i-PadSize+KernelSize,j-PadSize:j-PadSize+KernelSize]) 

        self.output = self.output.astype(np.uint8)
        cv2.imwrite('Max.png',self.output)
        self.show_output()


    def MedianFilter(self):
        KernelSize = 3 if self.ui.KernelSize.text()=="" else int(self.ui.KernelSize.text())
        PadSize= KernelSize//2
        PadImg = np.pad(self.img,((PadSize,PadSize),(PadSize,PadSize)),'edge')
        #print(PadImg)
        self.output = np.zeros((self.img.shape[0],self.img.shape[1]))


        for i in range(PadSize,PadImg.shape[0]-PadSize):
            for j in range(PadSize,PadImg.shape[1]-PadSize):                
                self.output[i-PadSize,j-PadSize] = 255-np.median(PadImg[i-PadSize:i-PadSize+KernelSize,j-PadSize:j-PadSize+KernelSize]) 
        
        #ret2,th2 = cv2.threshold(self.output.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.output = self.output.astype(np.uint8)
        cv2.imwrite('Med.png',self.output)
        self.show_output()
        







    def show_output(self):
        qimage_output = QImage(self.output, self.output.shape[1], self.output.shape[0],self.output.shape[1], QImage.Format_Grayscale8)

        self.ui.ImageShowOutput.setPixmap(QPixmap.fromImage(qimage_output))
        self.ui.ImageShowOutput.setScaledContents(True)                





if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())