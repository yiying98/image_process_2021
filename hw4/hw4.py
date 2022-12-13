import sys
from PyQt5.QtWidgets import *
from hw4qt import Ui_MainWindow
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
        self.ui.LoadButton.clicked.connect(self.load_data)
        self.ui.FFTButton.clicked.connect(self.FFT)
        self.ui.IFFTButton.clicked.connect(self.IFFT)
        self.ui.IdealLowButton.clicked.connect(self.IdealLow)
        self.ui.IdealHighButton.clicked.connect(self.IdealHigh)
        self.ui.ButterLowButton.clicked.connect(self.ButterLow)
        self.ui.ButterHighButton.clicked.connect(self.ButterHigh)
        self.ui.GaussiaLowButton.clicked.connect(self.GaussianLow)
        self.ui.GaussiaHighButton.clicked.connect(self.GaussianHigh)
        self.ui.HomoButton.clicked.connect(self.Homo)
        self.ui.MotionButton.clicked.connect(self.blur)
        self.ui.InverseButton.clicked.connect(self.Inverse)
        self.ui.WienerButton.clicked.connect(self.Wiener)
        self.ui.NoiseButton.clicked.connect(self.Noise)
        self.ui.D0Input.setText('30')
        self.ui.HighInput.setText('0.2')
        self.ui.LowInput.setText('2.2')
        self.output_1 = None
        self.output_2 = None
        self.d0 = None
        self.blur = None

        


    def load_data(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self,"開啟檔案","./","(*bmp *png);;(*.bmp *.png)") 
        if fileName1 != '':
            img = cv2.imread(fileName1)
            self.img = ((img[:,:,0]/3+img[:,:,1]/3+img[:,:,2]/3)).astype(np.uint8)
            qimage_output = QImage(self.img, self.img.shape[1], self.img.shape[0],self.img.shape[1], QImage.Format_Grayscale8)
            self.ui.ImageShow1.setPixmap(QPixmap.fromImage(qimage_output))
            self.ui.ImageShow1.setScaledContents(True)      



    def FFT(self):

        f = np.fft.fft2(self.img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(1+np.absolute(fshift))
        fmax = np.max(magnitude_spectrum)
        fmin = magnitude_spectrum.min()
        out =np.array((255*( magnitude_spectrum - fmin )/( fmax - fmin))).astype(np.uint8)
        out2 = np.array(225*(np.angle(np.fft.fftshift(f)))/(2*math.pi)).astype(np.uint8)
        self.output_1= out.copy()
        self.output_2 = out2.copy()
        self.show_output_1()
        self.show_output_2()

    def IFFT(self):

        f = np.fft.fft2(self.img)
        fshift = np.fft.fftshift(f)
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back).astype(np.uint8)
        self.output_1= img_back.copy()
        self.show_output_1()

    def IdealHigh(self):
        self.d0 = int(self.ui.D0Input.text())
        f = np.fft.fft2(self.img)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img.shape
        crow,ccol = int(rows/2), int(cols/2)
        #fshift[crow-int(self.d0/2):crow+int(self.d0/2), ccol-int(self.d0/2):ccol+int(self.d0/2)] = 0
        for i in range(rows):
            for j in range(cols):
                if math.sqrt((i-crow)**2+(j-ccol)**2)<self.d0:
                    fshift[i][j]=0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back).astype(np.uint8)
        self.output_1= img_back.copy()
        self.show_output_1()
    

    
    def IdealLow(self):
        self.d0 = int(self.ui.D0Input.text())
        f = np.fft.fft2(self.img)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img.shape
        crow,ccol = int(rows/2) , int(cols/2)
        for i in range(rows):
            for j in range(cols):
                if math.sqrt((i-crow)**2+(j-ccol)**2)>self.d0:
                    fshift[i][j]=0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back).astype(np.uint8)
        self.output_1= img_back.copy()
        self.show_output_1()

    def ButterHigh(self):
        self.d0 = int(self.ui.D0Input.text())
        f = np.fft.fft2(self.img)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img.shape
        crow,ccol = int(rows/2), int(cols/2)
        n = 2
        for i in range(rows):
            for j in range(cols):
                d = math.sqrt((i-crow)**2+(j-ccol)**2)
                h = 1/(1+(d/self.d0)**(2*n))
                fshift[i][j] = fshift[i][j] *(1-h)
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back).astype(np.uint8)
        self.output_1= img_back.copy()
        self.show_output_1()
    
    def ButterLow(self):
        self.d0 = int(self.ui.D0Input.text())
        f = np.fft.fft2(self.img)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img.shape
        crow,ccol = int(rows/2), int(cols/2)
        n = 2
        for i in range(rows):
            for j in range(cols):
                d = math.sqrt((i-crow)**2+(j-ccol)**2)
                h = 1/(1+(d/self.d0)**(2*n))
                fshift[i][j] = fshift[i][j] *h
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back).astype(np.uint8)
        self.output_1= img_back.copy()
        self.show_output_1()


    def GaussianHigh(self):
        self.d0 = int(self.ui.D0Input.text())
        f = np.fft.fft2(self.img)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img.shape
        crow,ccol = int(rows/2), int(cols/2)
        for i in range(rows):
            for j in range(cols):
                d = math.sqrt((i-crow)**2+(j-ccol)**2)
                h = math.exp(-d**2/(2*self.d0**2))
                fshift[i][j] = fshift[i][j] *(1-h)
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back).astype(np.uint8)
        self.output_1= img_back.copy()
        self.show_output_1()
    
    def GaussianLow(self):
        self.d0 = int(self.ui.D0Input.text())
        f = np.fft.fft2(self.img)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img.shape
        crow,ccol = int(rows/2), int(cols/2)
        for i in range(rows):
            for j in range(cols):
                d = math.sqrt((i-crow)**2+(j-ccol)**2)
                h = math.exp(-d**2/(2*self.d0**2))
                fshift[i][j] = fshift[i][j] *h
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back).astype(np.uint8)
        self.output_1= img_back.copy()
        self.show_output_1()


    def Homo(self):
        self.d0 =int(self.ui.D0Input.text())
        gl = float(self.ui.HighInput.text())
        gh = float(self.ui.LowInput.text())
        
        img =np.log(self.img+0.01)
        f = np.fft.fft2(self.img/225)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img.shape
        crow,ccol = int(rows/2), int(cols/2)
        
        for i in range(rows):
            for j in range(cols):
                d = (i-crow)**2+(j-ccol)**2
                h = (gh-gl)*(1-np.exp(-1*d/self.d0**2))+gl
                fshift[i][j] = fshift[i][j] *h
        
        fshift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(fshift)
        
        img_back = np.real(img_back)
        
        img_back = (np.exp(img_back)-0.01)
        #print(img_back)
        img_back =cv2.convertScaleAbs(img_back, alpha=255/img_back.max()).astype(np.uint8)
        #img_back = np.abs(img_back).astype(np.uint8)
        
        self.output_1= img_back.copy()
        self.show_output_1()
       
    def blur(self):
        f = np.fft.fft2(self.img)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img.shape
        crow,ccol = int(rows/2), int(cols/2)
        a=0.02
        b=0.02
        for i in range(rows):
            for j in range(cols):
                    uv = i*a+j*b+1e-6
                    h = 1/(math.pi*uv)*math.sin(math.pi*uv)*np.exp(-1j*math.pi*uv)
                    fshift[i][j] = fshift[i][j]*h

        
        fshift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(fshift)
        self.blur = img_back.copy()
        img_back = np.abs(img_back)
        img_back = cv2.convertScaleAbs(img_back, alpha=255/img_back.max()).astype(np.uint8)
        self.output_1= img_back.copy()
        self.show_output_1()
    

    def Wiener(self):
        f = np.fft.fft2(self.blur)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img.shape
        crow,ccol = int(rows/2), int(cols/2)
        a=0.02
        b=0.02
        k=1
        for i in range(rows):
            for j in range(cols):
                    uv = i*a+j*b+1e-6
                    h = 1/(math.pi*uv)*math.sin(math.pi*uv)*np.exp(-1j*math.pi*uv)
                    fshift[i][j] = fshift[i][j]*(1/h+(np.abs(h**2)/(20+np.abs(h**2))))
        
        fshift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(fshift)
        img_back = np.abs(img_back)
        img_back = cv2.convertScaleAbs(img_back, alpha=255/img_back.max()).astype(np.uint8)
        
        self.output_2= img_back.copy()
        self.show_output_2()       

    def Noise(self):
        img = self.img + np.random.normal(0 , 20, self.img.shape)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img.shape
        crow,ccol = int(rows/2), int(cols/2)
        a=0.1
        b=0.1
        for i in range(rows):
            for j in range(cols):
                    uv = i*a+j*b+1e-6
                    h = 1/(math.pi*uv)*math.sin(math.pi*uv)*np.exp(-1j*math.pi*uv)
                    fshift[i][j] = fshift[i][j]*h

        
        fshift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(fshift)
        self.blur = img_back.copy()
        img_back = np.abs(img_back)
        img_back = cv2.convertScaleAbs(img_back, alpha=255/img_back.max()).astype(np.uint8)
        img = cv2.convertScaleAbs(img, alpha=255/img.max()).astype(np.uint8)
        self.output_1= img_back.copy()
        self.show_output_1()
    








        
    def Inverse(self):
        f = np.fft.fft2(self.blur)
        fshift = np.fft.fftshift(f)
        rows, cols = self.img.shape
        crow,ccol = int(rows/2), int(cols/2)
        a=0.02
        b=0.02
        for i in range(rows):
            for j in range(cols):
                    uv = i*a+j*b+1e-6
                    h = 1/(math.pi*uv)*math.sin(math.pi*uv)*np.exp(-1j*math.pi*uv)
                    fshift[i][j] = fshift[i][j]/h
        
        fshift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(fshift)
        img_back = np.abs(img_back)
        img_back = cv2.convertScaleAbs(img_back, alpha=255/img_back.max()).astype(np.uint8)
        
        self.output_2= img_back.copy()
        self.show_output_2()        



    def show_output_1(self):
        h,w = self.output_1.shape

        qimage_output_1 = QImage(self.output_1, w, h,w, QImage.Format_Grayscale8)

        self.ui.ImageShow2.setPixmap(QPixmap.fromImage(qimage_output_1))
        self.ui.ImageShow2.setScaledContents(True)                

    def show_output_2(self):
        h,w = self.output_2.shape

        qimage_output_2 = QImage(self.output_2, w, h,w, QImage.Format_Grayscale8)

        self.ui.ImageShow2_2.setPixmap(QPixmap.fromImage(qimage_output_2))
        self.ui.ImageShow2_2.setScaledContents(True)                




if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())