import sys
from PyQt5.QtWidgets import *
from hw6qt import Ui_MainWindow
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re
import math
from scipy import signal
from PyQt5.QtGui import QImage, QPixmap, qRgb
import numpy as np
from utils import transform, linear, \
        transX, transY, scaleX, scaleY
from sklearn import cluster, mixture

harr_scale = np.array([1, 1]) / np.sqrt(2)
harr_wavel = np.array([1, -1]) / np.sqrt(2)
harr_scale_r = np.flip(harr_scale)
harr_wavel_r = np.flip(harr_wavel)


def convolve(data, f):
    """
    1D convolve function apply across 2D
    """
    return np.stack([np.convolve(d, f) for d in data])

def upSample(data):
    """ Upsameple the 1D array and add 0 between data """
    z = np.zeros([data.shape[0], data.shape[1] * 2])
    z[:, ::2] = data
    return z

def LOG(data):

    sig=math.sqrt(3)
    GaussianSize = 11
    ax = np.linspace(-(GaussianSize - 1) / 2., (GaussianSize - 1) / 2., GaussianSize)
    one = np.ones((len(ax),len(ax)))
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.multiply(one-(np.square(xx) + np.square(yy))/(2*np.square(sig)),np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig)))
    output = signal.convolve2d(data, kernel, boundary='symm', mode='same').astype(np.uint8)
    return output

def geoTransform(img, want_mask):
    """ Geometric transform """
    # Make mask same x shape with image
    x, y = np.where(want_mask)
    affine = scaleX(img.shape[0] / (x.max() - x.min() + 1)) * transX(-x.min())
    want_fullx = transform(want_mask, affine, new_shape=img.shape)
    # Resize the image by row
    img_pad = np.pad(img, [[0, 0], [0, 1]])
    want_fullx_img = np.zeros(want_fullx.shape)
    for i in range(img.shape[0]):
        padrow = img_pad[i]
        y = np.where(want_fullx[i])[0]
        corr_y = (y - y.min()) / (y.max() - y.min()) * (img.shape[1] - 1)
        int_y = corr_y.astype(np.int)
        want_fullx_img[i, y] = linear(corr_y,
                                      padrow[int_y], padrow[int_y + 1])

    # Inverse affine transform to original shape of mask
    want_img = transform(want_fullx_img, np.linalg.inv(affine))
    want_img = want_img[:want_mask.shape[0], :want_mask.shape[1]]
    return want_img



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.LoadButton1.clicked.connect(self.load_data1)
        self.ui.LoadButton2.clicked.connect(self.load_data2)
        self.ui.TrapezoidalButton.clicked.connect(self.Trap)
        self.ui.CIrcularButton.clicked.connect(self.Circular)
        self.ui.WavyButton.clicked.connect(self.Wavy)
        self.ui.DWTButton.clicked.connect(self.DWT)
        self.ui.HoughButton.clicked.connect(self.Hough)
        self.ui.kInput.setText('1')
        self.img1 = None
        self.imggray1 = None
        self.img2 = None
        self.imggray2 = None
        self.output_1 = None
        self.k = 1

    def load_data1(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self,"開啟檔案","./","(*bmp *jpg);;(*.bmp *.jpg)") 
        if fileName1 != '':
            img = cv2.imread(fileName1)
            self.img1 = img
            self.imggray1 = ((img[:,:,0]/3+img[:,:,1]/3+img[:,:,2]/3)).astype(np.uint8)
            qimage_output = QImage(self.imggray1, self.imggray1.shape[1], self.imggray1.shape[0],self.imggray1.shape[1] ,QImage.Format_Grayscale8).rgbSwapped()   
            self.ui.ImageShow1.setPixmap(QPixmap.fromImage(qimage_output))
            self.ui.ImageShow1.setScaledContents(True)   

    def load_data2(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self,"開啟檔案","./","(*bmp *jpg);;(*.bmp *.jpg)") 
        if fileName1 != '':
            img = cv2.imread(fileName1)
            self.img2 = img
            self.imggray2 = ((img[:,:,0]/3+img[:,:,1]/3+img[:,:,2]/3)).astype(np.uint8)
            qimage_output = QImage(self.imggray2, self.imggray2.shape[1], self.imggray2.shape[0],self.imggray2.shape[1] ,QImage.Format_Grayscale8).rgbSwapped()   
            self.ui.ImageShow2.setPixmap(QPixmap.fromImage(qimage_output))
            self.ui.ImageShow2.setScaledContents(True)   

    def show_output_1(self):
        h,w = self.output_1.shape

        qimage_output_1 = QImage(self.output_1, w, h,w, QImage.Format_Grayscale8)

        self.ui.ImageShow3.setPixmap(QPixmap.fromImage(qimage_output_1))
        self.ui.ImageShow3.setScaledContents(True) 

    def Trap(self):
        mask = np.ones([self.img1.shape[0],self.img1.shape[1]])
        y, x = np.meshgrid(np.arange(mask.shape[0]),np.arange(mask.shape[1]))
        mask[y<0.3*x]=0
        mask[(y-self.img1.shape[0])>-0.3*x]=0
        mask[x>self.img1.shape[1]*0.7]=0
        imggray = self.imggray1/255
        self.output_1 = (geoTransform(imggray,mask)*255).astype(np.uint8)
        self.show_output_1()
    
    def Circular(self):
        mask = np.zeros([self.img1.shape[0],self.img1.shape[1]])
        y, x = np.meshgrid(np.arange(mask.shape[0]),np.arange(mask.shape[1]))
        mask[(x-500)**2+(y-500)**2< 500 ** 2]=1
        imggray = self.imggray1/255
        self.output_1 = (geoTransform(imggray,mask)*255).astype(np.uint8)
        self.show_output_1()

    def Wavy(self):
        amp = 20
        T = 200
        y, x = np.meshgrid(np.arange(self.img1.shape[0]),np.arange(self.img1.shape[1]))
        x_trans = np.ceil(x+amp*np.sin(2*math.pi*y/T))
        y_trans = np.ceil(y+amp*np.sin(2*math.pi*x/T))
        print(x_trans)
        print(y_trans)
        img = self.imggray1
        output = np.zeros([self.img1.shape[0],self.img1.shape[1]])

        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                if x_trans[row][col] >=0 and y_trans[row][col] >=0 and x_trans[row][col] < self.img1.shape[1] and y_trans[row][col]< self.img1.shape[0] :
                    u = int(y_trans[row][col])
                    v= int(x_trans[row][col])
                    output[row][col] = img[v][u]
                else:
                    output[row][col] = 0

        self.output_1 = output.astype(np.uint8)
        self.show_output_1()
    


    def padTo2power(self,img):
        """ Make the shape of image to 2**n """
        s = np.array(2 ** np.ceil(np.log2(img.shape)), dtype=np.int)
        a = s - img.shape
        return np.pad(img, [[0, a[0]], [0, a[1]]])

    

    def wavelet2D(self,data, depth):
        """ Wavelet transform with Harr """
        if not depth:
            return data

        # by column
        scale = convolve(data.T, harr_scale_r)[:, 1::2].T
        wavel = convolve(data.T, harr_wavel_r)[:, 1::2].T

        # by row
        scale_scale = convolve(scale, harr_scale_r)[:, 1::2]
        wavel_h     = convolve(scale, harr_wavel_r)[:, 1::2]
        wavel_v     = convolve(wavel, harr_scale_r)[:, 1::2]
        wavel_wavel = convolve(wavel, harr_wavel_r)[:, 1::2]

        # recursion
        scale_scale = self.wavelet2D(scale_scale, depth - 1)
        return np.vstack([np.hstack([scale_scale, wavel_h]),
                        np.hstack([wavel_v, wavel_wavel])])

    def wavelet2DInv(self,data, depth):
        """ Inversed Wavelet transform with Harr """
        if not depth:
            return data
        h, w = np.array(data.shape) // 2

        # recursion
        scale_scale = self.wavelet2DInv(data[:h, :w], depth - 1)

        # by row
        scale_scale = convolve(upSample( scale_scale), harr_scale)[:, :-1]
        wave_h      = convolve(upSample(data[:h, w:]), harr_wavel)[:, :-1]
        wave_v      = convolve(upSample(data[h:, :w]), harr_scale)[:, :-1]
        wavel_wavel = convolve(upSample(data[h:, w:]), harr_wavel)[:, :-1]

        # by column
        scale = convolve(upSample((scale_scale + wave_h).T), harr_scale)[:, :-1].T
        wavel = convolve(upSample((wavel_wavel + wave_v).T), harr_wavel)[:, :-1].T
        return wavel + scale




    
    def DWT(self):
        self.kInput = int(self.ui.kInput.text())


        imgs = [self.imggray1/255, self.imggray2/255]
        ori_size = imgs[0].shape


        # wavelet
        imgs = [self.padTo2power(img) for img in imgs]
        wave_img = [self.wavelet2D(img, self.kInput) for img in imgs]

        # fusion (Compare abs value)
        subsize = np.array(imgs[0].shape) // 2 ** self.kInput
        minmax = np.min(wave_img, axis=0), np.max(wave_img, axis=0)
        v = np.argmax(np.abs(minmax), axis=0)
        new_img = minmax[1]
        new_img[v == 0] = minmax[0][v == 0]
        new_img[:subsize[0], :subsize[1]] = \
            np.mean(wave_img, axis=0)[:subsize[0], :subsize[1]]
        output =  self.wavelet2DInv(new_img, self.kInput)[:ori_size[0], :ori_size[1]]
        output = output*255
        self.output_1 = output.astype(np.uint8).copy()
        self.show_output_1()

    def Hough(self):
        edge = LOG(self.imggray1)
        
        #self.output_1 = edge.astype(np.uint8).copy()
        #self.show_output_1()
        
        edge = LOG(self.imggray1)/255 > .5
        edge = edge[30:470, 30:470]
        max_dis = np.ceil(np.sqrt(np.sum(np.array(edge.shape) ** 2))).astype(np.int) + 1

        # hough
        th = np.linspace(0, np.pi, 181)
        x, y = np.where(edge)
        r = (np.outer(x, np.cos(th)) + np.outer(y, np.sin(th))).astype(np.int)
        hough = np.zeros([2 * max_dis, th.size])
        for i in range(len(x)):
            hough[r[i] + max_dis, np.arange(th.size)] += 1

        # k-mean
        k = 8
        crit = 70
        hough_crit = hough.copy()
        hough_crit[hough < crit] = 0
        x, y = np.where(hough_crit)
        model = cluster.KMeans(k).fit(np.array([y, x]).T, sample_weight=hough_crit[x, y])
        centers = model.cluster_centers_
        centers = centers[centers.argsort(axis=0)[:, 0]]
        print(centers)


        # reverse
        x = np.arange(0, edge.shape[0])
        restore_edge = np.zeros(edge.shape)
        for i, center in enumerate(centers):
            th = center[0] / 180 * np.pi
            y = ((center[1] - max_dis - x * np.cos(th)) / np.sin(th)).astype(np.int)
            index_in = np.logical_and(y >= 0, y < edge.shape[1])
            restore_edge[x[index_in], y[index_in]] = 1
        
        self.output_1 = (restore_edge*225).astype(np.uint8).copy()
        self.show_output_1()











if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())