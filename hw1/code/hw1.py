import sys
from PyQt5.QtWidgets import *
from hw1QT import Ui_MainWindow
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage, QPixmap, qRgb


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.OpenFileButton.clicked.connect(self.load_data)
        self.ui.pushButton_3.clicked.connect(self.load_second_data)
        self.first_img = np.zeros((64,64))
        self.second_img = np.zeros((64,64))
        self.output_img = np.zeros((64,64))
        self.ui.AddSlider.valueChanged[int].connect(self.add)
        self.ui.AddSlider.setTickInterval(1)
        self.ui.AddSlider.setSingleStep(1)
        self.ui.MultiplySlider.valueChanged[int].connect(self.multiply)
        self.ui.MultiplySlider.setTickInterval(1)
        self.ui.MultiplySlider.setSingleStep(1)
        self.ui.AvergeButton.clicked.connect(self.average)
        self.ui.pushButton_2.clicked.connect(self.fooFunction)
        self.ui.ExitButton.clicked.connect(self.close)


    def load_data(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self,"開啟檔案","./","(*64);; (*.64)") 
        if fileName1 != '':
            with open(fileName1, "rb") as f:
                image = f.readlines()
                #convert value to list
                for index, line  in enumerate(image):
                    line = list(line)
                    image[index]=np.fromiter(map(lambda n: int(256*(n-48)/32) if n<58 else int(256*(n-55)/32) , line[:64]),
                     dtype=np.int8)
                
                #convert list to ndarray
                image_arr = np.array(image).astype(np.uint8)
                self.first_img = image_arr

                #plot histogram
                self.plot_input_histogram(self.first_img,'input.png')

                #convert array to Pixmap
                hist = cv2.imread('input.png')

                qimage_input = QImage(self.first_img.data, self.first_img.shape[0], self.first_img.shape[1], 
                self.first_img.shape[0], QImage.Format_Grayscale8)
                
                qimage_hist =QImage(hist.data, hist.shape[1], hist.shape[0], hist.shape[1]*3, 
                QImage.Format_RGB888).rgbSwapped()


                self.ui.SourceImageShow.setPixmap(QPixmap.fromImage(qimage_input))
                self.ui.SourceImageShow.setScaledContents(True)
                
                self.ui.HistogamShow.setPixmap(QPixmap.fromImage(qimage_hist))
                self.ui.HistogamShow.setScaledContents(True)


    def load_second_data(self):
        fileName2, filetype = QFileDialog.getOpenFileName(self,"開啟檔案","./","(*64);; (*.64)") 
        if fileName2 != '':
            with open(fileName2, "rb") as f:
                image = f.readlines()

                #convert value to list
                for index, line  in enumerate(image):
                    line = list(line)
                    image[index]= np.fromiter(map(lambda n: int(256*(n-48)/32) if n<58 else int(256*(n-56)/32) ,
                     line[:64]), dtype=np.int8)
                
                #convert list to ndarray
                image_arr = np.array(image).astype(np.uint8)
                self.second_img = image_arr


                qimage_input = QImage(image_arr.data, image_arr.shape[0], image_arr.shape[1], image_arr.shape[0],
                 QImage.Format_Grayscale8)


                self.ui.SecondImageShow.setPixmap(QPixmap.fromImage(qimage_input))
                self.ui.SecondImageShow.setScaledContents(True)


    def add(self,value):
        image = [item for subarray in self.first_img for item in subarray]
        image = np.fromiter(map(lambda n: (n+int(256*value/32)) if 0<=(n+int(256*value/32)) \
        and (n+int(256*value/32))<=255 else (255 if (n+int(256*value/32))>255 else 0), image), dtype=np.int8)
        
        self.output_img = np.reshape(image,(64,-1)).astype(np.uint8)

        self.plot_input_histogram(self.output_img ,'add.png')
        self.show_output('add.png')
    

    def multiply(self,value):
        image = [item for subarray in self.first_img for item in subarray]
        image = np.fromiter(map(lambda n: n*int(value) if 0<= n*int(value) and 0<= n*int(value)<=255 \
        else (255 if n*int(value)>255 else 0), image), dtype=np.int8)
        
        self.output_img = np.reshape(image,(64,-1)).astype(np.uint8)

        self.plot_input_histogram(self.output_img, 'multiply.png')
        self.show_output('multiply.png')
        

    def average(self):
        image1 = [item for subarray in self.first_img for item in subarray]
        image2 = [item for subarray in self.second_img for item in subarray]
        image = np.fromiter(map(lambda x,y: int(x/2+y/2), image1, image2), dtype=np.int8)
        self.output_img = np.reshape(image, (64,-1)).astype(np.uint8)

        self.plot_input_histogram(self.output_img,'average.png')
        self.show_output('average.png')

    def fooFunction(self):
        image1 = [item for subarray in self.first_img for item in subarray]
        image2 = [item for sublist in self.first_img for item in [0] + list(sublist[:-1])]
        image = np.fromiter(map(lambda x,y: int(x-y) if 0<=(int(x)-int(y)) \
        and (int(x)-int(y))<=255 else (0 if (int(x)-int(y))<0 else 255) , image1, image2), dtype=np.int8)
        
        self.output_img = np.reshape(image, (64,-1)).astype(np.uint8)

        self.plot_input_histogram(self.output_img,'fooFunction.png')
        self.show_output('fooFunction.png')

    def plot_input_histogram(self ,array ,name):
        plt.hist(array.ravel(), 32, [0, 256])
        plt.savefig(name)
        plt.close('all')


    def show_output(self ,hist_name):
        hist = cv2.imread(hist_name)
        qimage_output = QImage(self.output_img.data, self.output_img.shape[1], self.output_img.shape[0],
         self.output_img.shape[1], QImage.Format_Grayscale8)
        qimage_hist =QImage(hist.data, hist.shape[1], hist.shape[0], hist.shape[1]*3,
         QImage.Format_RGB888).rgbSwapped()
        self.ui.OutputImageShow.setPixmap(QPixmap.fromImage(qimage_output))
        self.ui.OutputImageShow.setScaledContents(True)                
        self.ui.OutputHistrogramShow_.setPixmap(QPixmap.fromImage(qimage_hist))
        self.ui.OutputHistrogramShow_.setScaledContents(True)
    



if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())