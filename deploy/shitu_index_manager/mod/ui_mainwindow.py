# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.5
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(833, 538)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.appMenuBtn = QtWidgets.QToolButton(self.centralwidget)
        self.appMenuBtn.setObjectName("appMenuBtn")
        self.horizontalLayout_3.addWidget(self.appMenuBtn)
        self.saveImageLibraryBtn = QtWidgets.QToolButton(self.centralwidget)
        self.saveImageLibraryBtn.setObjectName("saveImageLibraryBtn")
        self.horizontalLayout_3.addWidget(self.saveImageLibraryBtn)
        self.addClassifyBtn = QtWidgets.QToolButton(self.centralwidget)
        self.addClassifyBtn.setObjectName("addClassifyBtn")
        self.horizontalLayout_3.addWidget(self.addClassifyBtn)
        self.removeClassifyBtn = QtWidgets.QToolButton(self.centralwidget)
        self.removeClassifyBtn.setObjectName("removeClassifyBtn")
        self.horizontalLayout_3.addWidget(self.removeClassifyBtn)
        spacerItem = QtWidgets.QSpacerItem(40, 20,
                                           QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.imageScaleSlider = QtWidgets.QSlider(self.centralwidget)
        self.imageScaleSlider.setMaximumSize(QtCore.QSize(400, 16777215))
        self.imageScaleSlider.setMinimum(1)
        self.imageScaleSlider.setMaximum(8)
        self.imageScaleSlider.setPageStep(2)
        self.imageScaleSlider.setOrientation(QtCore.Qt.Horizontal)
        self.imageScaleSlider.setObjectName("imageScaleSlider")
        self.horizontalLayout_3.addWidget(self.imageScaleSlider)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.widget = QtWidgets.QWidget(self.splitter)
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.searchClassifyHistoryCmb = QtWidgets.QComboBox(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.searchClassifyHistoryCmb.sizePolicy().hasHeightForWidth())
        self.searchClassifyHistoryCmb.setSizePolicy(sizePolicy)
        self.searchClassifyHistoryCmb.setEditable(True)
        self.searchClassifyHistoryCmb.setObjectName("searchClassifyHistoryCmb")
        self.horizontalLayout.addWidget(self.searchClassifyHistoryCmb)
        self.searchClassifyBtn = QtWidgets.QToolButton(self.widget)
        self.searchClassifyBtn.setObjectName("searchClassifyBtn")
        self.horizontalLayout.addWidget(self.searchClassifyBtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.classifyListView = QtWidgets.QListView(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.classifyListView.sizePolicy().hasHeightForWidth())
        self.classifyListView.setSizePolicy(sizePolicy)
        self.classifyListView.setMinimumSize(QtCore.QSize(200, 0))
        self.classifyListView.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.classifyListView.setObjectName("classifyListView")
        self.verticalLayout_2.addWidget(self.classifyListView)
        self.widget1 = QtWidgets.QWidget(self.splitter)
        self.widget1.setObjectName("widget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.addImageBtn = QtWidgets.QToolButton(self.widget1)
        self.addImageBtn.setObjectName("addImageBtn")
        self.horizontalLayout_2.addWidget(self.addImageBtn)
        self.removeImageBtn = QtWidgets.QToolButton(self.widget1)
        self.removeImageBtn.setObjectName("removeImageBtn")
        self.horizontalLayout_2.addWidget(self.removeImageBtn)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20,
                                            QtWidgets.QSizePolicy.Expanding,
                                            QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.imageListWidget = QtWidgets.QListWidget(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.imageListWidget.sizePolicy().hasHeightForWidth())
        self.imageListWidget.setSizePolicy(sizePolicy)
        self.imageListWidget.setMinimumSize(QtCore.QSize(200, 0))
        self.imageListWidget.setStyleSheet(
            "QListWidget::Item:hover{background:skyblue;padding-top:0px; padding-bottom:0px;}\n"
            "QListWidget::item:selected{background:rgb(245, 121, 0); color:red;}"
        )
        self.imageListWidget.setObjectName("imageListWidget")
        self.verticalLayout.addWidget(self.imageListWidget)
        self.verticalLayout_3.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "识图图像库管理"))
        self.appMenuBtn.setText(_translate("MainWindow", "..."))
        self.saveImageLibraryBtn.setText(_translate("MainWindow", "..."))
        self.addClassifyBtn.setText(_translate("MainWindow", "..."))
        self.removeClassifyBtn.setText(_translate("MainWindow", "..."))
        self.searchClassifyBtn.setText(_translate("MainWindow", "..."))
        self.addImageBtn.setText(_translate("MainWindow", "..."))
        self.removeImageBtn.setText(_translate("MainWindow", "..."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
