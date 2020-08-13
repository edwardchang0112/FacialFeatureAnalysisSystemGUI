# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QBrush, QColor
import os
from FA_main import cross_validation_by_file, trainedModel, shapAnalysis
from Readtxt import readTXTFile

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1025, 812)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 10, 721, 61))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(430, 60, 321, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(110, 110, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(230, 110, 131, 27))
        self.pushButton.setObjectName("pushButton")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(110, 210, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(100, 240, 201, 27))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(340, 240, 171, 27))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(320, 210, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(560, 170, 181, 31))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(340, 270, 211, 61))
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(120, 410, 151, 27))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(110, 480, 161, 27))
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(110, 330, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(340, 280, 381, 51))
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(110, 140, 421, 61))
        self.listWidget.setObjectName("listWidget")
        self.listWidget_2 = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_2.setGeometry(QtCore.QRect(110, 270, 171, 61))
        self.listWidget_2.setObjectName("listWidget_2")
        self.listWidget_4 = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_4.setGeometry(QtCore.QRect(330, 460, 211, 51))
        self.listWidget_4.setObjectName("listWidget_4")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(110, 360, 171, 51))
        self.textEdit.setObjectName("textEdit")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(110, 520, 171, 20))
        self.checkBox.setObjectName("checkBox")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(110, 440, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setItalic(True)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(290, 520, 171, 20))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setGeometry(QtCore.QRect(110, 550, 171, 20))
        self.checkBox_3.setObjectName("checkBox_3")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(400, 650, 151, 27))
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(550, 510, 431, 51))
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.listWidget_5 = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_5.setGeometry(QtCore.QRect(120, 570, 181, 71))
        self.listWidget_5.setObjectName("listWidget_5")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(310, 650, 91, 27))
        self.pushButton_7.setObjectName("pushButton_7")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(550, 570, 281, 71))
        self.label_13.setText("")
        self.label_13.setObjectName("label_13")
        self.listWidget_6 = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_6.setGeometry(QtCore.QRect(330, 570, 211, 71))
        self.listWidget_6.setObjectName("listWidget_6")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(340, 360, 381, 51))
        self.label_12.setText("")
        self.label_12.setObjectName("label_12")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1025, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        ####################################################
        self.clickedTrainedModel = str()
        self.clickedNewModel = str()
        self.DataSet_FileList = []
        self.chooseModelName = str()
        self.chooseTwoFeatureSet = set()
        self.model = QtGui.QStandardItemModel()
        self.pushButton.clicked.connect(self.browseDataSet)
        self.pushButton_2.clicked.connect(self.browseNewModel)
        self.pushButton_3.clicked.connect(self.runCrossValidation)
        self.pushButton_4.clicked.connect(self.trainNewModel)
        self.pushButton_5.clicked.connect(self.browseTrainedModel)
        self.pushButton_6.clicked.connect(self.featureAnalysis)
        self.pushButton_7.clicked.connect(self.clearChosenFeatures)

        self.checkBox_3.stateChanged.connect(self.showTwoFeatures)

        self.listWidget_2.itemClicked.connect(self.clickNewModel)
        self.listWidget_4.itemClicked.connect(self.clickTrainedModel)
        self.listWidget_5.itemClicked.connect(self.twoFeatureClicked)

    def browseDataSet(self):
        self.label_6.clear()
        self.label_6.repaint()
        self.listWidget.clear()  # initial item list in listWidget
        cwd = os.getcwd()  # get current parent directory path
        # cwd = cwd+'/P2_Use_Skincare/Use_Skincare/'
        print("cwd = ", cwd)

        files, ok1 = QFileDialog.getOpenFileNames(None, "Choose Multiple files", cwd, r'Excel Files(*.xlsx)')
        if len(files) == 0:
            print("\nCancel")
            return
        print("\nChoose Files:")
        for file in files:
            print(file)
        print(files, ok1)
        self.DataSet_FileList = files
        # slm = QtCore.QStringListModel()
        self.listWidget.addItems(files)
        # self.listWidget.itemClicked.connect(self.listwidgetclicked)
        # self.listView.selectionModel().currentChanged.connect(self.on_treeView_clicked)
        self.label_6.setText("Total Numbers of Files: " + str(len(files)))
        self.label_6.repaint()

    def clickNewModel(self, item):
        print('click -> {}'.format(item.text()))
        self.clickedNewModel = str(item.text())
        print("self.clickedNewModel = ", self.clickedNewModel)

    def clickTrainedModel(self, item):
        print('click -> {}'.format(item.text()))
        self.clickedTrainedModel = str(item.text())
        print("self.clickedTrainedModel = ", self.clickedTrainedModel)

    def twoFeatureClicked(self, item):
        print('click -> {}'.format(item.text()))
        chooseTwoFeatureSet_old = self.chooseTwoFeatureSet.copy()
        self.chooseTwoFeatureSet.add(str(item.text()))
        print("self.chooseTwoFeatureSet - chooseTwoFeatureSet_old = ",
              self.chooseTwoFeatureSet - chooseTwoFeatureSet_old)
        if len(self.chooseTwoFeatureSet - chooseTwoFeatureSet_old) > 0:
            self.listWidget_6.addItems(list(self.chooseTwoFeatureSet - chooseTwoFeatureSet_old))
        else:
            pass

    def browseNewModel(self):
        self.listWidget_2.clear()  # initial item list in listWidget_2
        cwd = os.getcwd()  # get current parent directory path
        cwd = cwd + '/New_Models/'
        print("cwd = ", cwd)
        # file, ok1 = QFileDialog.getOpenFileName(None, "Choose Multiple files", cwd, r'Text Files(*.txt)')
        # if len(file) == 0:
        #    print("\nCancel")
        #    return
        # print("\nChoose Files:")
        # print(file, ok1)
        file_name = "New_Models"
        files = readTXTFile(cwd + file_name)
        self.listWidget_2.addItems(files)

    def runCrossValidation(self):
        self.label_9.setText("")
        self.label_9.repaint()
        chooseModel = self.clickedNewModel
        model_name = chooseModel.split(".")[0].split("/")[-1]
        # model_name = str(model_name)
        print("model_name = ", model_name)
        if len(self.DataSet_FileList) == 0:
            self.label_9.setText("Please back to the last step, \nmake sure that at least 1 Dataset is chosen")
            self.label_9.repaint()
        else:
            self.label_9.setText("")
            self.label_9.repaint()
            error_mean_rms_hydration_all_mean, error_std_rms_hydration_all_mean, error_mean_rms_oxygen_all_mean, \
            error_std_rms_oxygen_all_mean = cross_validation_by_file(self.DataSet_FileList, model_name)

            self.label_8.setText("rms_error_hydration: " + str(error_mean_rms_hydration_all_mean)[:5] + "%\n"
                                 + "rms_error_oxygen: " + str(error_mean_rms_oxygen_all_mean)[:5] + "%")
            self.label_8.repaint()

        self.textEdit.setText(model_name)
        self.textEdit.repaint()

    def trainNewModel(self):
        self.label_12.setText("")
        self.label_12.repaint()
        stored_path = 'Trained_Models/'
        trainedModelName = self.textEdit.toPlainText()
        trainedModelName = str(trainedModelName)
        if len(trainedModelName) != 0:
            if len(self.DataSet_FileList) != 0:
                trainedModel(self.DataSet_FileList, trainedModelName, stored_path)
            else:
                self.label_12.setText(
                    "Please go back to the 'Data Set Selection' step, \nmake sure choose 1 existing data set at least!")
                self.label_12.repaint()
        else:
            self.label_12.setText("Please follow 'Model Selection' step first, \nthen start to train a model!")
            self.label_12.repaint()

    def browseTrainedModel(self):
        self.listWidget_4.clear()
        self.clickedTrainedModel = str()
        cwd = os.getcwd()  # get current parent directory path
        cwd = cwd + '/Trained_Models/'
        print("cwd = ", cwd)
        files, ok1 = QFileDialog.getOpenFileNames(None, "Choose Multiple files", cwd, r'All Files(*.*)')
        file_names = []
        if len(files) == 0:
            print("\nCancel")
            return
        print("\nChoose Files:")
        for file in files:
            file = str(file).split("/")[-1]
            file_names.append(file)
            print(file)
        print(files, ok1)
        self.listWidget_4.addItems(file_names)

    def showTwoFeatures(self):
        if self.checkBox_3.isChecked():
            cwd = os.getcwd()  # get current parent directory path
            cwd_NM = cwd + '/Feature_Names/'
            file_name = "Feature_Names"
            features_list = readTXTFile(cwd_NM + file_name)
            self.listWidget_5.addItems(features_list)
        else:
            self.listWidget_5.clear()

    def clearChosenFeatures(self):
        self.listWidget_6.clear()
        self.chooseTwoFeatureSet = set()

    def featureAnalysis(self):
        self.label_13.setText("")
        self.label_13.repaint()
        cwd = os.getcwd()  # get current parent directory path
        cwd_NM = cwd + '/Feature_Names/'
        file_name = "Feature_Names"
        features_list = readTXTFile(cwd_NM + file_name)
        print("features_list = ", features_list)
        #choose_features = self.listWidget_6.currentItem()
        choose_features = []
        for index in range(self.listWidget_6.count()):
            choose_features.append(self.listWidget_6.item(index).text())
        print("choose_features = ", choose_features)
        features_list_index = []
        for feature in choose_features:
            features_list_index.append(features_list.index(feature))
            print("features_list_index = ", features_list_index)

        cwd_TM = cwd + '/Trained_Models/'
        model_path = cwd_TM
        print("self.clickedTrainedModel = ", self.clickedTrainedModel)
        chooseModel = self.clickedTrainedModel
        model_name = chooseModel

        if len(model_name) == 0:
            self.label_11.setText("Please go back to the last step, \nmake sure choose 1 existing model at least!")
            self.label_11.repaint()
        elif len(self.DataSet_FileList) == 0:
            self.label_11.setText(
                "Please go back to the 'Data Set Selection' step, \nmake sure choose 1 existing data set at least!")
            self.label_11.repaint()
        else:
            self.label_11.setText("")
            self.label_11.repaint()
            if (not self.checkBox.isChecked()) and (not self.checkBox_2.isChecked()) and (
                    not self.checkBox_3.isChecked()):
                self.label_13.setText("Please choose 1 feature analysis method at least!")
                self.label_13.repaint()
            else:
                self.label_13.setText("")
                self.label_13.repaint()
                if self.checkBox.isChecked():
                    shapAnalysis(model_path + model_name, features_list, None, self.DataSet_FileList,
                                 str(self.checkBox.text()))
                if self.checkBox_2.isChecked():
                    shapAnalysis(model_path + model_name, features_list, None, self.DataSet_FileList,
                                 str(self.checkBox_2.text()))
                if self.checkBox_3.isChecked():
                    if len(list(self.chooseTwoFeatureSet)) == 2:
                        shapAnalysis(model_path + model_name, features_list, features_list_index, self.DataSet_FileList,
                                     str(self.checkBox_3.text()))
                    else:
                        self.label_13.setText("Please re-choose '2 Features'!")
                        self.label_13.repaint()

    ####################################################

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "AI Algorithm Performance Validation and Feature Analysis"))
        self.label_2.setText(_translate("MainWindow", "Ver. 1.0 Developed by Edward Chang 20200211"))
        self.label_3.setText(_translate("MainWindow", "Data Set Selection"))
        self.pushButton.setText(_translate("MainWindow", "Choose Data Set"))
        self.label_4.setText(_translate("MainWindow", "Model Selection"))
        self.pushButton_2.setText(_translate("MainWindow", "Browse New Model Methods"))
        self.pushButton_3.setText(_translate("MainWindow", "Run Performance Test"))
        self.label_5.setText(_translate("MainWindow", "Performance Result"))
        self.pushButton_4.setText(_translate("MainWindow", "Train Model"))
        self.pushButton_5.setText(_translate("MainWindow", "Browse Trained Model"))
        self.label_7.setText(_translate("MainWindow", "Model Training"))
        self.checkBox.setText(_translate("MainWindow", "Overall features analysis"))
        self.label_10.setText(_translate("MainWindow", "Feature Analysis"))
        self.checkBox_2.setText(_translate("MainWindow", "Cross feature analysis"))
        self.checkBox_3.setText(_translate("MainWindow", "Two-Feature analysis"))
        self.pushButton_6.setText(_translate("MainWindow", "Run Feature Analysis"))
        self.pushButton_7.setText(_translate("MainWindow", "Re-choose"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
