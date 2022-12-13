# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw4.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1181, 510)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.LoadButton = QtWidgets.QPushButton(self.centralwidget)
        self.LoadButton.setGeometry(QtCore.QRect(50, 10, 113, 32))
        self.LoadButton.setObjectName("LoadButton")
        self.FFTButton = QtWidgets.QPushButton(self.centralwidget)
        self.FFTButton.setGeometry(QtCore.QRect(180, 10, 113, 32))
        self.FFTButton.setObjectName("FFTButton")
        self.IFFTButton = QtWidgets.QPushButton(self.centralwidget)
        self.IFFTButton.setGeometry(QtCore.QRect(310, 10, 113, 32))
        self.IFFTButton.setObjectName("IFFTButton")
        self.IdealLowButton = QtWidgets.QPushButton(self.centralwidget)
        self.IdealLowButton.setGeometry(QtCore.QRect(180, 60, 113, 32))
        self.IdealLowButton.setObjectName("IdealLowButton")
        self.ButterLowButton = QtWidgets.QPushButton(self.centralwidget)
        self.ButterLowButton.setGeometry(QtCore.QRect(440, 60, 151, 32))
        self.ButterLowButton.setObjectName("ButterLowButton")
        self.GaussiaLowButton = QtWidgets.QPushButton(self.centralwidget)
        self.GaussiaLowButton.setGeometry(QtCore.QRect(760, 60, 131, 32))
        self.GaussiaLowButton.setObjectName("GaussiaLowButton")
        self.D0Input = QtWidgets.QLineEdit(self.centralwidget)
        self.D0Input.setGeometry(QtCore.QRect(330, 120, 61, 32))
        self.D0Input.setObjectName("D0Input")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(330, 100, 21, 16))
        self.label_3.setObjectName("label_3")
        self.HomoButton = QtWidgets.QPushButton(self.centralwidget)
        self.HomoButton.setGeometry(QtCore.QRect(180, 120, 131, 32))
        self.HomoButton.setObjectName("HomoButton")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(420, 100, 81, 16))
        self.label_4.setObjectName("label_4")
        self.LowInput = QtWidgets.QLineEdit(self.centralwidget)
        self.LowInput.setGeometry(QtCore.QRect(420, 120, 61, 32))
        self.LowInput.setObjectName("LowInput")
        self.HighInput = QtWidgets.QLineEdit(self.centralwidget)
        self.HighInput.setGeometry(QtCore.QRect(520, 120, 61, 32))
        self.HighInput.setText("")
        self.HighInput.setObjectName("HighInput")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(520, 100, 111, 16))
        self.label_5.setObjectName("label_5")
        self.MotionButton = QtWidgets.QPushButton(self.centralwidget)
        self.MotionButton.setGeometry(QtCore.QRect(180, 170, 131, 32))
        self.MotionButton.setObjectName("MotionButton")
        self.InverseButton = QtWidgets.QPushButton(self.centralwidget)
        self.InverseButton.setGeometry(QtCore.QRect(320, 170, 131, 32))
        self.InverseButton.setObjectName("InverseButton")
        self.NoiseButton = QtWidgets.QPushButton(self.centralwidget)
        self.NoiseButton.setGeometry(QtCore.QRect(580, 170, 131, 32))
        self.NoiseButton.setObjectName("NoiseButton")
        self.ImageShow1 = QtWidgets.QLabel(self.centralwidget)
        self.ImageShow1.setGeometry(QtCore.QRect(10, 230, 351, 291))
        self.ImageShow1.setText("")
        self.ImageShow1.setObjectName("ImageShow1")
        self.ImageShow2 = QtWidgets.QLabel(self.centralwidget)
        self.ImageShow2.setGeometry(QtCore.QRect(410, 230, 351, 291))
        self.ImageShow2.setText("")
        self.ImageShow2.setObjectName("ImageShow2")
        self.ImageShow2_2 = QtWidgets.QLabel(self.centralwidget)
        self.ImageShow2_2.setGeometry(QtCore.QRect(790, 230, 351, 291))
        self.ImageShow2_2.setText("")
        self.ImageShow2_2.setObjectName("ImageShow2_2")
        self.IdealHighButton = QtWidgets.QPushButton(self.centralwidget)
        self.IdealHighButton.setGeometry(QtCore.QRect(310, 60, 113, 32))
        self.IdealHighButton.setObjectName("IdealHighButton")
        self.ButterHighButton = QtWidgets.QPushButton(self.centralwidget)
        self.ButterHighButton.setGeometry(QtCore.QRect(600, 60, 151, 32))
        self.ButterHighButton.setObjectName("ButterHighButton")
        self.GaussiaHighButton = QtWidgets.QPushButton(self.centralwidget)
        self.GaussiaHighButton.setGeometry(QtCore.QRect(900, 60, 131, 32))
        self.GaussiaHighButton.setObjectName("GaussiaHighButton")
        self.WienerButton = QtWidgets.QPushButton(self.centralwidget)
        self.WienerButton.setGeometry(QtCore.QRect(450, 170, 131, 32))
        self.WienerButton.setObjectName("WienerButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1181, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.LoadButton.setText(_translate("MainWindow", "Load Image"))
        self.FFTButton.setText(_translate("MainWindow", "FFT"))
        self.IFFTButton.setText(_translate("MainWindow", "IFFT"))
        self.IdealLowButton.setText(_translate("MainWindow", "Ideal Low"))
        self.ButterLowButton.setText(_translate("MainWindow", "Butterworth Low"))
        self.GaussiaLowButton.setText(_translate("MainWindow", "Gaussian Low"))
        self.label_3.setText(_translate("MainWindow", "D0"))
        self.HomoButton.setText(_translate("MainWindow", "Homomorphic"))
        self.label_4.setText(_translate("MainWindow", "Gamma Low"))
        self.label_5.setText(_translate("MainWindow", "Gamma High"))
        self.MotionButton.setText(_translate("MainWindow", "Motion Blur"))
        self.InverseButton.setText(_translate("MainWindow", "Inverse FIlter"))
        self.NoiseButton.setText(_translate("MainWindow", "Gaussian noise"))
        self.IdealHighButton.setText(_translate("MainWindow", "Ideal HIgh"))
        self.ButterHighButton.setText(_translate("MainWindow", "Butterworth High"))
        self.GaussiaHighButton.setText(_translate("MainWindow", "Gaussian High"))
        self.WienerButton.setText(_translate("MainWindow", "Wiener Filter"))