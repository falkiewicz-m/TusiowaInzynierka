import cv2
import math
import numpy as np
import os
from PyQt5 import QtCore, QtGui, QtWidgets
import sqlite3
from matplotlib import pyplot as plt
import sys


# Parametry programu

# Zmienne globalne
#####################################
# środek źrenicy
pupilCenter = (0, 0)
# promień źrenicy
pupilRadius = 0
# promień tęczówki
irisRadius = 0


#####################################

conn = sqlite3.connect('baza.db')
c = conn.cursor()

# Wczytuje obraz, znajduje na nim źrenicę, zapisuje współrzędne jej środka
# w zmiennych globalnych, zamalowuje obrys źrenicy na czarno i zwraca
# obraz.

# @param image      Obraz do obróbki
# @returns image    Obraz z zamalowaną źrenicą
def getPupil(frame):
    global minPupilSize, pupilCenter, pupilRadius
    pupilImg = frame.copy()
    print(frame.shape)
    # progowanie - zostawiamy tylko bardzo ciemne fragmenty obrazu
    frameThresh = cv2.inRange(frame, (0, 0, 0), (80, 80, 80))
    # cv2.imshow('prethresh',frameThresh)

    # szukanie konturów
    _, contours, hierarchy = cv2.findContours(frameThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

    # poglądowe wyrysowanie znalezionych konturów
    cv2.drawContours(pupilImg, contours, -1, (0, 255, 255), 2)
    #cv2.imshow('out', pupilImg)

    del pupilImg
    pupilImg = frame.copy()
    # iterujemy po konturach i szukamy takiego, o największej powierzchni
    c_max = contours[0]
    area_max = 0
    for c in contours:
        # znajdź momenty konturu
        moments = cv2.moments(c)
        # moment m00 jest powierzchnią konturu
        area = moments['m00']
        # print("Current area: ",  area)
        # jeśli powierzchnia jest większa niż ostatnia
        if (area > area_max):
            c_max = c
            area_max = area
            # wylicz współrzędne na podstawie momentu m10 i m01
            x = int(moments['m10'] / area)
            y = int(moments['m01'] / area)

            # zapisz współrzędne środka
            pupilCenter = (int(x), int(y))
    # zamaluj największy kontur na czarno - poglądowo
    # drawContours w drugim parametrze oczekuje tablicy konturów, a nie pojedynczego konturu
    # stąd też jest sztuczne opakowanie obecnego konturu w tablicę
    cv2.drawContours(pupilImg, [c_max], -1, (0, 0, 0), -1)

    # ze wzoru na pole koła przybliżamy promień źrenicy
    pupilRadius = np.round(np.sqrt(area_max / np.pi)).astype('int')

    return (pupilImg)


# Wykrywa tęczówkę i przycina obraz tak by tylko ona była widoczna.
# Zaczernia wszystko poza okręgiem wykrytym jako tęczówka.
#
# @param image        Image with black-painted pupil
# @returns image     Image with isolated iris + black-painted pupil
def getIris(frame):
    # cv2.imshow('in', frame)
    resImg = frame.copy()
    grayImg = frame.copy()
    grayImg = cv2.cvtColor(grayImg, cv2.COLOR_BGR2GRAY)
    minIN = 30
    maxIN = 200
    minOUT = 0
    maxOUT = 255
    extendedHist = grayImg.copy()
    a = (maxOUT - minOUT)/(maxIN-minIN)
    b = minOUT - a*minIN
    for i in range(grayImg.shape[0]):
        for j in range(grayImg.shape[1]):
            extendedHist[i,j] = np.round(i*a + b)
            if extendedHist[i,j] < 0:
                extendedHist[i,j] = 0
            continue
            if extendedHist[i,j] > 255:
                extendedHist[i,j] = 255
            continue
        break
    

    grayImg = extendedHist.copy()
    # cv2.imshow('grayImg', grayImg)

    irisCircle = getCircles(grayImg)

    if (irisCircle.shape[0] == 3):
        # średnica tęczówki
        rad = int(irisCircle[2])
        # print(rad)
        global irisRadius
        irisRadius = rad

        # rysowanie maskowanego oka
        # wraz ze współrzędnymi kwadratu do przycięcia
        cv2.circle(grayImg, pupilCenter, rad, (255, 255, 255), cv2.FILLED)
        cv2.circle(grayImg, (pupilCenter[0] - rad, pupilCenter[1] - rad), 1, (0, 0, 0), cv2.FILLED)
        cv2.circle(grayImg, (pupilCenter[0] - rad, pupilCenter[1] + rad), 1, (0, 0, 0), cv2.FILLED)
        cv2.circle(grayImg, (pupilCenter[0] + rad, pupilCenter[1] + rad), 1, (0, 0, 0), cv2.FILLED)
        cv2.circle(grayImg, (pupilCenter[0] + rad, pupilCenter[1] - rad), 1, (0, 0, 0), cv2.FILLED)
        # cv2.imshow('masked', grayImg);

        # maska - okrąg o promieniu tęczówki o srodku w srodku źrenicy
        mask = np.full((frame.shape[0], frame.shape[1]), 0, dtype=np.uint8)
        cv2.circle(mask, pupilCenter, rad, (255, 255, 255), cv2.FILLED)

        # wytnij wszystko zgodnie z maską
        maskedFrame = cv2.bitwise_and(resImg, resImg, mask=mask)
        # cv2.imshow('masked', maskedFrame)

        # przycięcie obrazu
        # kwadrat o boku rad*2
        # o środku w środku źrenicy
        x = int(pupilCenter[0] - rad)
        y = int(pupilCenter[1] - rad)
        w = int(rad * 2)

        resImg = maskedFrame[y:y + w, x:x + w:1]

    return resImg


# Wykrywa okręgi na zdjęciu przy pomocy transformaty Hough.
# Przeszukuje przestrzeń parametrów transformaty Hough do momentu,
# aż znajdzie tylko jeden okrąg.
#
# Zwraca znaleziony okrąg lub pustą listę w razie porażki.
#
# @param image - zdjęcie
# @returns tuple -
def getCircles(image):
    i = 50
    # zakładamy, ze źrenica jest mniejsza niż najmniejszy wymiar zdjęcia (mieści się w zdjęciu)
    maxCircleRadius = np.round((min(image.shape[0], image.shape[1]) / 2) * 0.9).astype("int")
    # zakładamy, że średnica/promień źrenicy to co najmniej 1/4 najmniejszego wymiaru
    minCircleRadius = np.round(min(image.shape[0], image.shape[1]) / 4).astype("int")
    while i < 150:
        print("HoughCircles - current param2: ", i)
        # HoughCircles
        # minDist - odległość między środkami znalezionych okręgów. zależy nam na znalezieniu tylko jednego.
        # minRadius - min. promień znalezionych okręgów
        # maxRadius - maks. promień znalezionych okręgow
        # param1, param2 - parametry algorytmu Hougha. Przeszukujemy przestrzeń parametru param2 od 1 do 150
        # wyznaczone eksperymentalnie
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=40, param2=i,
                                   minRadius=minCircleRadius, maxRadius=maxCircleRadius)

        # jeśli znaleziono
        if circles is not None:
            # zaokrąglanie współrzędnych
            circles = np.round(circles[0, :]).astype("int")

        i += 1
    return np.zeros((1))


# wyliczanie współrzędnych kartezjańskich z polarnych
def polar2cart(r, theta, center):
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return x, y


def img2polar(img, center, final_radius, initial_radius=None, phase_width=1000):
    if initial_radius is None:
        initial_radius = 0

    theta, R = np.meshgrid(np.linspace(0, 2 * np.pi, phase_width),
                           np.arange(initial_radius, final_radius))

    Xcart, Ycart = polar2cart(R, theta, center)

    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)

    if img.ndim == 3:
        polar_img = img[Xcart, Ycart, :]
        polar_img = np.reshape(polar_img, (np.round(final_radius - initial_radius).astype('int'), phase_width, 3))
    else:
        polar_img = img[Xcart, Ycart]
        polar_img = np.reshape(polar_img, (np.round(final_radius - initial_radius).astype('int'), phase_width))

    return polar_img


# Zamienia obraz będący wyciętą tęczówką z czarną źrenicą w środku
# zgodnie z przekształceniem współrzędnych biegunowych na kartezjańskie
#
# @param image      Zdjęcie tęczówki
# @returns image    Przekształcone zdjęcie

def getPolar2CartImg(image):
    global pupilRadius, irisRadius
    # zakładamy, że mamy kwadratowe zdjecie a źrenica jest dokładnie w srodku
    imgSize = image.shape[0]
    c = (float(imgSize / 2.0), float(imgSize / 2.0))
    # szerokość paska to promień tęczowki minus promień źrenicy
    width = irisRadius - pupilRadius

    imgRes = np.full((width, image.shape[1]), 0, dtype=np.uint8)

    # cv2.LogPolar(image,imgRes,c,50.0, cv2.CV_INTER_LINEAR+cv2.CV_WARP_FILL_OUTLIERS)
    cv2.LogPolar(image, imgRes, c, 60.0, cv2.CV_INTER_LINEAR + cv2.CV_WARP_FILL_OUTLIERS)
    return (imgRes)


cv2.startWindowThread();


####### INTERFEJS UŻYTKOWNIKA #######


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(701, 502)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 10, 691, 81))
        font = QtGui.QFont()
        font.setFamily("Adobe Caslon Pro Bold")
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.Select_Image = QtWidgets.QPushButton(self.centralwidget)
        self.Select_Image.setGeometry(QtCore.QRect(10, 390, 231, 71))
        font = QtGui.QFont()
        font.setFamily("Adobe Caslon Pro Bold")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.Select_Image.setFont(font)
        self.Select_Image.setStyleSheet("font: 75 12pt \"Adobe Caslon Pro Bold\";\n"
                                        "background-color: rgb(30, 195, 255);\n"
                                        "border-radius: 12px;\n"
                                        "hover: {background-color: #3e8e41}\n"
                                        "active: {\n"
                                        "  background-color: #3e8e41;\n"
                                        "  box-shadow: 0 5px #666;\n"
                                        "  transform: translateY(4px);}\n"
                                        "\n"
                                        "")
        self.Select_Image.setObjectName("Select_Image")
        self.Select_Image.clicked.connect(self.setImage)

        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(120, 110, 471, 251))
        self.imageLbl.setText("")
        self.imageLbl.setObjectName("imageLbl")
        self.Analize_Image = QtWidgets.QPushButton(self.centralwidget)
        self.Analize_Image.setGeometry(QtCore.QRect(460, 390, 231, 71))
        font = QtGui.QFont()
        font.setFamily("Adobe Caslon Pro Bold")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(9)
        self.Analize_Image.setFont(font)
        self.Analize_Image.setStyleSheet("font: 75 12pt \"Adobe Caslon Pro Bold\";\n"
                                         "background-color: rgb(30, 195, 255);\n"
                                         "border-radius: 12px;\n"
                                         "hover: {background-color: #3e8e41}\n"
                                         "active: {\n"
                                         "  background-color: #3e8e41;\n"
                                         "  box-shadow: 0 5px #666;\n"
                                         "  transform: translateY(4px);}\n"
                                         "\n"
                                         "")
        self.Analize_Image.setObjectName("Analize_Image")
        self.Analize_Image.clicked.connect(self.Analize)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 701, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "System Biometryczny"))
        self.label.setText(_translate("MainWindow",
                                      "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">SYSTEM BIOMETRYCZNY</span></p><p align=\"center\"><span style=\" font-size:16pt;\">BAZUJĄCY NA ANALIZIE WZORU TĘCZÓWKI OKA</span></p></body></html>"))
        self.Select_Image.setText(_translate("MainWindow", "WYBIERZ ZDJĘCIE"))
        self.Analize_Image.setText(_translate("MainWindow", "ANALIZUJ"))

    def setImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.bmp)")
        if fileName:
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio)
            pixmap.save("eye.bmp")
            self.imageLbl.setPixmap(pixmap)
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)
            print(fileName)


    def Analize(self):

        key = 0

        # wczytaj zdjęcie
        frame = cv2.imread("eye.bmp", cv2.COLOR_BGR2GRAY)



        # znajdź źrenicę
        frameBlackPupil = getPupil(frame)

        frameBlackPupilCenter = frameBlackPupil.copy()
        cv2.circle(frameBlackPupilCenter, pupilCenter, 3, (255,255,255), cv2.FILLED)
        cv2.circle(frameBlackPupilCenter, pupilCenter, pupilRadius, (255,255,255), 1)

        cv2.imshow('WYKRYTA ZRENICA', frameBlackPupilCenter)

        #jeśli znaleziono źrenicę
        if(pupilCenter != (0,0)):
        # znajdź tęczówkę
            iris = getIris(frameBlackPupil)
            cv2.imshow('WYKRYTA TECZOWKA',iris)
            normImg = iris.copy()
            normImg = img2polar(iris, (iris.shape[0]/2, iris.shape[1]/2), iris.shape[0]/2, pupilRadius)
            cv2.imshow("NORMALIZACJA", normImg)
        cv2.waitKey(0)


        cv2.destroyAllWindows()

if __name__ == "__main__":


    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
