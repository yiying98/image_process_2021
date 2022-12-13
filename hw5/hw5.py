import sys
from PyQt5.QtWidgets import *
from hw5qt import Ui_MainWindow
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re
import math
from scipy import signal
from PyQt5.QtGui import QImage, QPixmap, qRgb
import numpy as np
from sklearn.cluster import KMeans



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.LoadButton.clicked.connect(self.load_data)
        self.ui.RGBButton.clicked.connect(self.to_RGB)
        self.ui.CMYButton.clicked.connect(self.to_CMY)
        self.ui.HSIButton.clicked.connect(self.to_HSI)
        self.ui.XYZButton.clicked.connect(self.to_XYZ)
        self.ui.LabButton.clicked.connect(self.to_Lab)
        self.ui.YUVButton.clicked.connect(self.to_YUV)
        self.ui.GrayScaleButton.clicked.connect(self.to_gray)
        self.ui.PseudoColorButton.clicked.connect(self.to_pcolor)
        self.ui.StartButton.clicked.connect(self.startcolor)
        self.ui.EndButton.clicked.connect(self.endcolor)
        self.ui.RGBButton_2.clicked.connect(self.k_RGB)
        self.ui.CMYButton_2.clicked.connect(self.k_CMY)
        self.ui.HSIButton_2.clicked.connect(self.k_HSI)
        self.ui.XYZButton_2.clicked.connect(self.k_XYZ)
        self.ui.LabButton_2.clicked.connect(self.k_Lab)
        self.ui.YUVButton_2.clicked.connect(self.k_YUV)
        self.ui.kInput.setText('2')
        self.img = None
        self.imggray = None
        self.output_1 = None
        self.output_2 = None
        self.output_3 = None
        self.start = np.array([0,0,0])
        self.end = np.array([255,255,255])
        self.k = 2

        
    def startcolor(self):
        color = QColorDialog.getColor() # OpenColorDialog
        if color.isValid():
            self.start = np.array([color.red(), color.green(), color.blue()])
            

    def endcolor(self):
        color = QColorDialog.getColor() # OpenColorDialog
        if color.isValid():
            self.end = np.array([color.red(), color.green(), color.blue()])
            

    def load_data(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self,"開啟檔案","./","(*bmp *png);;(*.bmp *.png)") 
        if fileName1 != '':
            img = cv2.imread(fileName1)
            self.img = img
            self.imggray = ((img[:,:,0]/3+img[:,:,1]/3+img[:,:,2]/3)).astype(np.uint8)
            qimage_output = QImage(self.img, self.img.shape[1], self.img.shape[0],self.img.shape[1]*3 ,QImage.Format_RGB888).rgbSwapped()   
            self.ui.ImageShow1.setPixmap(QPixmap.fromImage(qimage_output))
            self.ui.ImageShow1.setScaledContents(True)      



    def to_RGB(self):
        im_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.output_1 = im_rgb[:,:,0].astype(np.uint8)
        self.output_2 = im_rgb[:,:,1].astype(np.uint8)
        self.output_3 = im_rgb[:,:,2].astype(np.uint8)
        self.show_output_1()
        self.show_output_2()
        self.show_output_3()


    def to_CMY(self):
        im_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        im_CMY = 255-im_rgb
        self.output_1 = im_CMY[:,:,0].astype(np.uint8)
        self.output_2 = im_CMY[:,:,1].astype(np.uint8)
        self.output_3 = im_CMY[:,:,2].astype(np.uint8)
        self.show_output_1()
        self.show_output_2()
        self.show_output_3()



    def to_HSI(self):
        im_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        i = np.mean(im_rgb, axis=2)
        s = 1 - np.min(im_rgb, axis=2) / (i + 1e-6)
        s[s < 0] = 0
        R, G, B = im_rgb[:, :, 0]/255, im_rgb[:, :, 1]/255, im_rgb[:, :, 2]/255
        th = np.arccos((2 * R - G - B) / 2 / (
             np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6))
        h = th
        h[B > G] = 2 * np.pi - h[B > G]
        h = cv2.convertScaleAbs(h,alpha=255/h.max())
        s = cv2.convertScaleAbs(s,alpha=255/s.max())
        i = cv2.convertScaleAbs(i,alpha=255/i.max())
        

        self.output_1 = h.astype(np.uint8)
        self.output_2 = s.astype(np.uint8)
        self.output_3 = i.astype(np.uint8)
        self.show_output_1()
        self.show_output_2()
        self.show_output_3()


    def to_XYZ(self):
        im_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)/255
        m = np.array([[0.412453, 0.357580, 0.180423 ],[0.212671, 0.715160, 0.072169],[0.019334 ,0.119193 ,0.950227]])
        im_xyz = im_rgb.dot(m.T)

        self.output_1 = (im_xyz[:,:,0]*255).astype(np.uint8)
        self.output_2 = (im_xyz[:,:,1]*255).astype(np.uint8)
        self.output_3 = (im_xyz[:,:,2]*255).astype(np.uint8)
        self.show_output_1()
        self.show_output_2()
        self.show_output_3()

    def to_Lab(self):
        im_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)/255
        m = np.array([[0.412453, 0.357580, 0.180423 ],[0.212671, 0.715160, 0.072169],[0.019334 ,0.119193 ,0.950227]])
        im_xyz = im_rgb.dot(m.T)
        w = np.array([0.3127,0.3290,0.3582])
        xyz = im_xyz/w
        i = xyz>0.008856
        xyz[i] = xyz[i]**(1/3)
        xyz[~i] = xyz[~i]*7.787+16/116 

        L = 116*xyz[:,:,1]-16
        a = 500*(xyz[:,:,0]-xyz[:,:,1]) 
        b = 200*(xyz[:,:,1]-xyz[:,:,2])

        L = cv2.convertScaleAbs(L,alpha=255/L.max())
        a = cv2.convertScaleAbs(a,alpha=255/a.max())
        b = cv2.convertScaleAbs(b,alpha=255/b.max())

        self.output_1 = L.astype(np.uint8)
        self.output_2 = a.astype(np.uint8)
        self.output_3 = b.astype(np.uint8)
        self.show_output_1()
        self.show_output_2()
        self.show_output_3()

    def to_YUV(self):
        m = np.array(np.matrix("""
                0.299 0.587 0.114;
                -0.14713 -0.28886 0.436;
                0.615 -0.51499 -0.10001"""))
        im_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)/255
        im_YUV = im_rgb.dot(m.T)

        Y,U,V = im_YUV[:,:,0],im_YUV[:,:,1],im_YUV[:,:,2]

        Y = cv2.convertScaleAbs(Y,alpha=255/Y.max())
        U = cv2.convertScaleAbs(U,alpha=255/U.max())
        V = cv2.convertScaleAbs(V,alpha=255/V.max())

        self.output_1 = Y.astype(np.uint8)
        self.output_2 = U.astype(np.uint8)
        self.output_3 = V.astype(np.uint8)
        self.show_output_1()
        self.show_output_2()
        self.show_output_3()


    def to_gray(self):
        im_gray =  cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.output_1 = im_gray.astype(np.uint8)
        self.show_output_1()
        self.ui.ImageShow3.clear()
        self.ui.ImageShow4.clear()
        
    def to_pcolor(self):
        im_gray =  cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)/255
        
        m = np.array([[0.412453, 0.357580, 0.180423 ],[0.212671, 0.715160, 0.072169],[0.019334 ,0.119193 ,0.950227]])
        start_xyz = (self.start/255).dot(m.T)
        end_xyz = (self.end/255).dot(m.T)
        inverse = np.linalg.inv(m)

        cmap = np.zeros([256, 3])
        
        
        for i in range(3):
            cmap[:, i] = np.linspace(start_xyz[i], end_xyz[i], 256)

        ind = np.array(im_gray * (cmap.shape[0] - 1) / np.max(im_gray)).astype(np.uint8)
        
        plt.figure()
        im = plt.imshow(ind,cmap=ListedColormap(cmap))
        plt.colorbar(im)
        plt.savefig("pcolor.png")

        img = cv2.imread("pcolor.png")
        
        qimage_output = QImage(img, img.shape[1], img.shape[0],img.shape[1]*3 ,QImage.Format_RGB888).rgbSwapped()   
        self.ui.ImageShow2.setPixmap(QPixmap.fromImage(qimage_output))
        self.ui.ImageShow2.setScaledContents(True)  
        self.ui.ImageShow3.clear()
        self.ui.ImageShow4.clear()


    def k_RGB(self):
        self.k = int(self.ui.kInput.text())
        im_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.kmeans(im_rgb)

    def k_CMY(self):
        self.k = int(self.ui.kInput.text())
        im_cmy = 1-cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.kmeans(im_cmy)

    def k_HSI(self):
        im_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        i = np.mean(im_rgb, axis=2)
        s = 1 - np.min(im_rgb, axis=2) / (i + 1e-6)
        s[s < 0] = 0
        R, G, B = im_rgb[:, :, 0]/255, im_rgb[:, :, 1]/255, im_rgb[:, :, 2]/255
        th = np.arccos((2 * R - G - B) / 2 / (
             np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6))
        h = th
        h[B > G] = 2 * np.pi - h[B > G]
        h = cv2.convertScaleAbs(h,alpha=255/h.max())
        s = cv2.convertScaleAbs(s,alpha=255/s.max())
        i = cv2.convertScaleAbs(i,alpha=255/i.max())
        im_hsi = np.stack([h, s, i], axis=2)
        self.kmeans(im_hsi)   

    def k_XYZ(self):
        im_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        m = np.array([[0.412453, 0.357580, 0.180423 ],[0.212671, 0.715160, 0.072169],[0.019334 ,0.119193 ,0.950227]])
        im_xyz = im_rgb.dot(m.T)
        self.kmeans(im_xyz)


    def k_Lab(self):
        im_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        m = np.array([[0.412453, 0.357580, 0.180423 ],[0.212671, 0.715160, 0.072169],[0.019334 ,0.119193 ,0.950227]])
        im_xyz = im_rgb.dot(m.T)
        w = np.array([0.3127,0.3290,0.3582])
        xyz = im_xyz/w
        i = xyz>0.008856
        xyz[i] = xyz[i]**(1/3)
        xyz[~i] = xyz[~i]*7.787+16/116 

        L = 116*xyz[:,:,1]-16
        a = 500*(xyz[:,:,0]-xyz[:,:,1]) 
        b = 200*(xyz[:,:,1]-xyz[:,:,2])

        L = cv2.convertScaleAbs(L,alpha=255/L.max())
        a = cv2.convertScaleAbs(a,alpha=255/a.max())
        b = cv2.convertScaleAbs(b,alpha=255/b.max())

        im_lab = np.stack([L, a, b], axis=2)
        self.kmeans(im_lab)

    def k_YUV(self):
        m = np.array(np.matrix("""
                0.299 0.587 0.114;
                -0.14713 -0.28886 0.436;
                0.615 -0.51499 -0.10001"""))
        im_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)/255
        im_YUV = im_rgb.dot(m.T)

        Y,U,V = im_YUV[:,:,0],im_YUV[:,:,1],im_YUV[:,:,2]

        Y = cv2.convertScaleAbs(Y,alpha=255/Y.max())
        U = cv2.convertScaleAbs(U,alpha=255/U.max())
        V = cv2.convertScaleAbs(V,alpha=255/V.max())
        im_yuv = np.stack([Y, U, V], axis=2)
        self.kmeans(im_yuv)


    def kmeans(self,inputpic):   
        x = inputpic.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(x)
        y_kmeans = kmeans.predict(x).reshape(inputpic.shape[0],-1)
        output = np.zeros(inputpic.shape)
        for i in range(self.k):
            f = y_kmeans==i
            output[f]=kmeans.cluster_centers_[i] 
        output = output.astype(np.uint8)
        qimage_output = QImage(output, output.shape[1], output.shape[0],output.shape[1]*3 ,QImage.Format_RGB888)  
        self.ui.ImageShow2.setPixmap(QPixmap.fromImage(qimage_output))
        self.ui.ImageShow2.setScaledContents(True)  
        self.ui.ImageShow3.clear()
        self.ui.ImageShow4.clear()
            
        






        


                


    def show_output_1(self):
        h,w = self.output_1.shape

        qimage_output_1 = QImage(self.output_1, w, h,w, QImage.Format_Grayscale8)

        self.ui.ImageShow2.setPixmap(QPixmap.fromImage(qimage_output_1))
        self.ui.ImageShow2.setScaledContents(True)                

    def show_output_2(self):
        h,w = self.output_2.shape

        qimage_output_2 = QImage(self.output_2, w, h,w, QImage.Format_Grayscale8)

        self.ui.ImageShow3.setPixmap(QPixmap.fromImage(qimage_output_2))
        self.ui.ImageShow3.setScaledContents(True)                

    def show_output_3(self):
        h,w = self.output_3.shape

        qimage_output_3 = QImage(self.output_3, w, h,w, QImage.Format_Grayscale8)

        self.ui.ImageShow4.setPixmap(QPixmap.fromImage(qimage_output_3))
        self.ui.ImageShow4.setScaledContents(True)                



if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())