# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw1.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1156, 776)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.OpenFileButton = QtWidgets.QPushButton(self.centralwidget)
        self.OpenFileButton.setGeometry(QtCore.QRect(70, 20, 113, 32))
        self.OpenFileButton.setObjectName("OpenFileButton")
        self.ExitButton = QtWidgets.QPushButton(self.centralwidget)
        self.ExitButton.setGeometry(QtCore.QRect(190, 20, 113, 32))
        self.ExitButton.setObjectName("ExitButton")
        self.SourceImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.SourceImageLabel.setGeometry(QtCore.QRect(80, 60, 91, 16))
        self.SourceImageLabel.setObjectName("SourceImageLabel")
        self.HistogamLabel = QtWidgets.QLabel(self.centralwidget)
        self.HistogamLabel.setGeometry(QtCore.QRect(360, 60, 91, 16))
        self.HistogamLabel.setObjectName("HistogamLabel")
        self.SourceImageShow = QtWidgets.QLabel(self.centralwidget)
        self.SourceImageShow.setGeometry(QtCore.QRect(80, 90, 261, 231))
        self.SourceImageShow.setText("")
        self.SourceImageShow.setObjectName("SourceImageShow")
        self.HistogamShow = QtWidgets.QLabel(self.centralwidget)
        self.HistogamShow.setGeometry(QtCore.QRect(360, 90, 261, 231))
        self.HistogamShow.setText("")
        self.HistogamShow.setObjectName("HistogamShow")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(740, 20, 113, 32))
        self.pushButton_3.setObjectName("pushButton_3")
        self.SecondImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.SecondImageLabel.setGeometry(QtCore.QRect(750, 60, 91, 16))
        self.SecondImageLabel.setObjectName("SecondImageLabel")
        self.SecondImageShow = QtWidgets.QLabel(self.centralwidget)
        self.SecondImageShow.setGeometry(QtCore.QRect(750, 90, 261, 231))
        self.SecondImageShow.setText("")
        self.SecondImageShow.setObjectName("SecondImageShow")
        self.ToolGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.ToolGroupBox.setGeometry(QtCore.QRect(60, 440, 341, 271))
        self.ToolGroupBox.setObjectName("ToolGroupBox")
        self.AddSlider = QtWidgets.QSlider(self.ToolGroupBox)
        self.AddSlider.setGeometry(QtCore.QRect(40, 50, 160, 22))
        self.AddSlider.setMinimum(-10)
        self.AddSlider.setMaximum(10)
        self.AddSlider.setOrientation(QtCore.Qt.Horizontal)
        self.AddSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.AddSlider.setObjectName("AddSlider")
        self.Addlabel = QtWidgets.QLabel(self.ToolGroupBox)
        self.Addlabel.setGeometry(QtCore.QRect(10, 50, 21, 16))
        self.Addlabel.setObjectName("Addlabel")
        self.label_2 = QtWidgets.QLabel(self.ToolGroupBox)
        self.label_2.setGeometry(QtCore.QRect(10, 100, 21, 16))
        self.label_2.setObjectName("label_2")
        self.MultiplySlider = QtWidgets.QSlider(self.ToolGroupBox)
        self.MultiplySlider.setGeometry(QtCore.QRect(40, 100, 160, 22))
        self.MultiplySlider.setMinimum(1)
        self.MultiplySlider.setMaximum(10)
        self.MultiplySlider.setOrientation(QtCore.Qt.Horizontal)
        self.MultiplySlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.MultiplySlider.setObjectName("MultiplySlider")
        self.AvergeButton = QtWidgets.QPushButton(self.ToolGroupBox)
        self.AvergeButton.setGeometry(QtCore.QRect(50, 150, 113, 32))
        self.AvergeButton.setObjectName("AvergeButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.ToolGroupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(50, 210, 113, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.ToolGroupBox)
        self.label.setGeometry(QtCore.QRect(180, 220, 131, 20))
        self.label.setObjectName("label")
        self.OutputImageShow = QtWidgets.QLabel(self.centralwidget)
        self.OutputImageShow.setGeometry(QtCore.QRect(450, 450, 261, 231))
        self.OutputImageShow.setText("")
        self.OutputImageShow.setObjectName("OutputImageShow")
        self.SecondImageLabel_2 = QtWidgets.QLabel(self.centralwidget)
        self.SecondImageLabel_2.setGeometry(QtCore.QRect(450, 420, 91, 16))
        self.SecondImageLabel_2.setObjectName("SecondImageLabel_2")
        self.OutputHistrogramShow_ = QtWidgets.QLabel(self.centralwidget)
        self.OutputHistrogramShow_.setGeometry(QtCore.QRect(810, 450, 261, 231))
        self.OutputHistrogramShow_.setText("")
        self.OutputHistrogramShow_.setObjectName("OutputHistrogramShow_")
        self.SecondImageLabel_3 = QtWidgets.QLabel(self.centralwidget)
        self.SecondImageLabel_3.setGeometry(QtCore.QRect(810, 420, 101, 16))
        self.SecondImageLabel_3.setObjectName("SecondImageLabel_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1156, 24))
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
        self.OpenFileButton.setText(_translate("MainWindow", "OpenFile"))
        self.ExitButton.setText(_translate("MainWindow", "Exit"))
        self.SourceImageLabel.setText(_translate("MainWindow", "Source Image"))
        self.HistogamLabel.setText(_translate("MainWindow", "Histogam"))
        self.pushButton_3.setText(_translate("MainWindow", "Second Image"))
        self.SecondImageLabel.setText(_translate("MainWindow", "Second Image"))
        self.ToolGroupBox.setTitle(_translate("MainWindow", "Tool"))
        self.Addlabel.setText(_translate("MainWindow", "+/-"))
        self.label_2.setText(_translate("MainWindow", "x"))
        self.AvergeButton.setText(_translate("MainWindow", "Averge"))
        self.pushButton_2.setText(_translate("MainWindow", "Function"))
        self.label.setText(_translate("MainWindow", "g(x,y)=f(x,y)-f(x-1,y)"))
        self.SecondImageLabel_2.setText(_translate("MainWindow", "Output Image"))
        self.SecondImageLabel_3.setText(_translate("MainWindow", "Output Histogam"))
