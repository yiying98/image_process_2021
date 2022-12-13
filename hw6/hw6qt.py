# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw6.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1042, 524)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.LoadButton1 = QtWidgets.QPushButton(self.centralwidget)
        self.LoadButton1.setGeometry(QtCore.QRect(10, 10, 121, 32))
        self.LoadButton1.setObjectName("LoadButton1")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 150, 60, 16))
        self.label.setObjectName("label")
        self.ImageShow1 = QtWidgets.QLabel(self.centralwidget)
        self.ImageShow1.setGeometry(QtCore.QRect(30, 190, 311, 331))
        self.ImageShow1.setText("")
        self.ImageShow1.setObjectName("ImageShow1")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(270, 10, 161, 31))
        self.label_5.setObjectName("label_5")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(270, 90, 111, 31))
        self.label_7.setObjectName("label_7")
        self.HoughButton = QtWidgets.QPushButton(self.centralwidget)
        self.HoughButton.setGeometry(QtCore.QRect(440, 90, 131, 32))
        self.HoughButton.setObjectName("HoughButton")
        self.ImageShow3 = QtWidgets.QLabel(self.centralwidget)
        self.ImageShow3.setGeometry(QtCore.QRect(670, 190, 311, 331))
        self.ImageShow3.setText("")
        self.ImageShow3.setObjectName("ImageShow3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(670, 150, 60, 16))
        self.label_2.setObjectName("label_2")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(440, 10, 280, 32))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.splitter_2 = QtWidgets.QSplitter(self.splitter)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.TrapezoidalButton = QtWidgets.QPushButton(self.splitter_2)
        self.TrapezoidalButton.setObjectName("TrapezoidalButton")
        self.WavyButton = QtWidgets.QPushButton(self.splitter_2)
        self.WavyButton.setObjectName("WavyButton")
        self.CIrcularButton = QtWidgets.QPushButton(self.splitter_2)
        self.CIrcularButton.setObjectName("CIrcularButton")
        self.splitter_3 = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_3.setGeometry(QtCore.QRect(270, 50, 221, 32))
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")
        self.splitter_5 = QtWidgets.QSplitter(self.splitter_3)
        self.splitter_5.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_5.setObjectName("splitter_5")
        self.label_6 = QtWidgets.QLabel(self.splitter_5)
        self.label_6.setObjectName("label_6")
        self.DWTButton = QtWidgets.QPushButton(self.splitter_5)
        self.DWTButton.setObjectName("DWTButton")
        self.ImageShow2 = QtWidgets.QLabel(self.centralwidget)
        self.ImageShow2.setGeometry(QtCore.QRect(350, 190, 311, 331))
        self.ImageShow2.setText("")
        self.ImageShow2.setObjectName("ImageShow2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(350, 150, 60, 16))
        self.label_4.setObjectName("label_4")
        self.splitter_4 = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_4.setGeometry(QtCore.QRect(510, 50, 162, 21))
        self.splitter_4.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_4.setObjectName("splitter_4")
        self.label_3 = QtWidgets.QLabel(self.splitter_4)
        self.label_3.setObjectName("label_3")
        self.kInput = QtWidgets.QLineEdit(self.splitter_4)
        self.kInput.setObjectName("kInput")
        self.LoadButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.LoadButton2.setGeometry(QtCore.QRect(10, 40, 121, 32))
        self.LoadButton2.setObjectName("LoadButton2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1042, 24))
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
        self.LoadButton1.setText(_translate("MainWindow", "Load 1st Image"))
        self.label.setText(_translate("MainWindow", "Input"))
        self.label_5.setText(_translate("MainWindow", "Geometric Transformation"))
        self.label_7.setText(_translate("MainWindow", "Hough Transform"))
        self.HoughButton.setText(_translate("MainWindow", "Hough Transform"))
        self.label_2.setText(_translate("MainWindow", "Output"))
        self.TrapezoidalButton.setText(_translate("MainWindow", "Trapezoidal"))
        self.WavyButton.setText(_translate("MainWindow", "Wavy"))
        self.CIrcularButton.setText(_translate("MainWindow", "Circular"))
        self.label_6.setText(_translate("MainWindow", "Image Fusion"))
        self.DWTButton.setText(_translate("MainWindow", "DWT"))
        self.label_4.setText(_translate("MainWindow", "Input2"))
        self.label_3.setText(_translate("MainWindow", "Depth"))
        self.LoadButton2.setText(_translate("MainWindow", "Load 2nd Image"))
