# importing libraries
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import pickle
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
import cv2
from PyQt5.QtWidgets import QMessageBox



# base prediction function
def predict_class(input_img,CNN_model,SVM_model):
	x_cnn=[]
	x_cnn.append(input_img)
	x_cnn=np.array(x_cnn)

	feature_extractor=CNN_model.predict(x_cnn)
	features = feature_extractor.reshape(feature_extractor.shape[0], -1)

	x_svm = features

	return SVM_model.predict(x_svm)[0]

# window class
class Window(QMainWindow):
	def __init__(self):
		super().__init__()

		# setting title
		self.setWindowTitle("Hindi Digit Recognition")

		# setting geometry to main window
		self.setGeometry(100, 100, 320, 320)

		# creating image object
		self.image = QImage(self.size(), QImage.Format_RGB32)

		# making image color to white↕
		self.image.fill(Qt.black)

		# variables
		# drawing flag
		self.drawing = False
		# default brush size
		self.brushSize = 20
		# default color
		self.brushColor = Qt.white

		# QPoint object to tract the point
		self.lastPoint = QPoint()

		# creating menu bar
		mainMenu = self.menuBar()

		# creating file menu for save and clear action
		fileMenu = mainMenu.addMenu("File")

		#adding prediction functions
		predict=mainMenu.addMenu("Predict")

		# creating save action
		saveAction = QAction("Save", self)
		# adding short cut for save action
		saveAction.setShortcut("Ctrl + S")
		# adding save to the file menu
		fileMenu.addAction(saveAction)
		# adding action to the save
		saveAction.triggered.connect(self.save)

		# creating clear action
		clearAction = QAction("Clear", self)
		# adding short cut to the clear action
		clearAction.setShortcut("Ctrl + C")
		# adding clear to the file menu
		fileMenu.addAction(clearAction)
		# adding action to the clear
		clearAction.triggered.connect(self.clear)

		# creating options for prediction
		# creating actions for different models
		ResNet = QAction("ResNet50_SVM", self)
		predict.addAction(ResNet)
		ResNet.triggered.connect(self.ResNet50_SVM)

		VGG16 = QAction("VGG16_SVM", self)
		predict.addAction(VGG16)
		VGG16.triggered.connect(self.VGG16_SVM)

		VGG19 = QAction("VGG19_SVM", self)
		predict.addAction(VGG19)
		VGG19.triggered.connect(self.VGG19_SVM)

	# method for checking mouse cicks
	def mousePressEvent(self, event):

		# if left mouse button is pressed
		if event.button() == Qt.LeftButton:
			# make drawing flag true
			self.drawing = True
			# make last point to the point of cursor
			self.lastPoint = event.pos()

	# method for tracking mouse activity
	def mouseMoveEvent(self, event):
		
		# checking if left button is pressed and drawing flag is true
		if (event.buttons() & Qt.LeftButton) & self.drawing:
			
			# creating painter object
			painter = QPainter(self.image)
			
			# set the pen of the painter
			painter.setPen(QPen(self.brushColor, self.brushSize,
							Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
			
			# draw line from the last point of cursor to the current point
			# this will draw only one step
			painter.drawLine(self.lastPoint, event.pos())
			
			# change the last point
			self.lastPoint = event.pos()
			# update
			self.update()

	# method for mouse left button release
	def mouseReleaseEvent(self, event):

		if event.button() == Qt.LeftButton:
			# make drawing flag false
			self.drawing = False

	# paint event
	def paintEvent(self, event):
		# create a canvas
		canvasPainter = QPainter(self)
		
		# draw rectangle on the canvas
		canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

	# method for saving canvas
	def save(self):
		filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
						"PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

		if filePath == "":
			return
		self.image.save(filePath)

	# method for clearing every thing on canvas
	def clear(self):
		# make the whole canvas white
		self.image.fill(Qt.black)
		# update
		self.update()



	# methods for prediction
	def ResNet50_SVM(self):
		# load trained SVM model
		filename = 'ResNet50_SVM_model.sav'

		SVM_model = pickle.load(open(filename, 'rb'))

		# load CNN for feature extraction
		CNN_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

		for layer in CNN_model.layers:
			layer.trainable = False
		
		text, okPressed = QInputDialog.getText(self, "Get text","File Path:", QLineEdit.Normal, "")
		path=text

		img = cv2.imread(path)
		img = cv2.resize(img, (32, 32),interpolation = cv2.INTER_AREA)
		# cv2.imwrite("UP_"+path,img)

		result=str(predict_class(img,CNN_model,SVM_model))
		msg = QMessageBox()
		msg.setWindowTitle("Prediction Output")
		msg.setText("Class: "+result)
		x = msg.exec_()



	def VGG16_SVM(self):
		# load trained SVM model
		filename = 'VGG16_SVM_model.sav'

		SVM_model = pickle.load(open(filename, 'rb'))

		# load CNN for feature extraction
		CNN_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

		for layer in CNN_model.layers:
			layer.trainable = False
		
		text, okPressed = QInputDialog.getText(self, "Get text","File Path:", QLineEdit.Normal, "")
		path=text

		img = cv2.imread(path)
		img = cv2.resize(img, (32, 32),interpolation = cv2.INTER_AREA)
		# cv2.imwrite("UP_"+path,img)

		result=str(predict_class(img,CNN_model,SVM_model))
		msg = QMessageBox()
		msg.setWindowTitle("Prediction Output")
		msg.setText("Class: "+result)
		x = msg.exec_()

	def VGG19_SVM(self):
		# load trained SVM model
		filename = 'VGG19_SVM_model.sav'

		SVM_model = pickle.load(open(filename, 'rb'))

		# load CNN for feature extraction
		CNN_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

		for layer in CNN_model.layers:
			layer.trainable = False
		
		text, okPressed = QInputDialog.getText(self, "Get text","File Path:", QLineEdit.Normal, "")
		path=text

		img = cv2.imread(path)
		img = cv2.resize(img, (32, 32),interpolation = cv2.INTER_AREA)
		# cv2.imwrite("UP_"+path,img)

		result=str(predict_class(img,CNN_model,SVM_model))
		msg = QMessageBox()
		msg.setWindowTitle("Prediction Output")
		msg.setText("Class: "+result)
		x = msg.exec_()



# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# showing the wwindow
window.show()

# start the app
sys.exit(App.exec())