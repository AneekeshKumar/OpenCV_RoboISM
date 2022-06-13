import math
import cv2 as cv
import numpy as np
from cv2 import aruco
import imutils

img_cv = cv.imread('CVtask.jpg')

imgGray = cv.cvtColor(img_cv, cv.COLOR_BGR2GRAY)
_, thrash = cv.threshold(imgGray, 240, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

cv.imshow("img", img_cv)
# a = 0
Aruco = [0, 0, 0, 0]
Aruco[0] = cv.imread('Ha.jpg')
Aruco[1] = cv.imread('HaHa.jpg')
Aruco[2] = cv.imread('XD.jpg')
Aruco[3] = cv.imread('LMAO.jpg')
colour = ""
corrected = [0, 0, 0, 0]
anything = []


def ArucoMarkers(img1):
    imgGray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    arucoDictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_250)
    arucoParameters = cv.aruco.DetectorParameters_create()

    Corners, Identifier, Rejected = cv.aruco.detectMarkers(imgGray, arucoDictionary, parameters=arucoParameters)
    Corners = np.array(Corners)
    Corners.resize(4, 2)

    # print(Corners)

    img2 = imutils.rotate(img1, 180 / math.pi * math.atan(
        (int(Corners[1][1]) - int(Corners[0][1])) / (int(Corners[1][0]) - int(Corners[0][0]))), scale=0.5)

    cv.imshow(f"{Identifier[0][0]}", img1)

    Corners, Identifier, Rejected = cv.aruco.detectMarkers(img2, arucoDictionary, parameters=arucoParameters)
    Corners = np.array(Corners)
    Corners.resize(4, 2)
    img2 = img2[int(Corners[0][0]):int(Corners[2][0]), int(Corners[0][1]): int(Corners[2][1])]
    corrected[Identifier[0][0]-1] = np.array(img2)
    cv.imshow(f"{Identifier[0][0]}", img2)
    return np.array(img2)


for x in range(4):
    anything.append(ArucoMarkers(Aruco[x]))
    # ArucoMarkers(Aruco(x))

for contour in contours:
    epsilon = 0.1 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 4:
        (x1, y1, w, h) = cv.boundingRect(approx)
        aspectRatio = float(w) / h
        print(x1, y1)
        if 0.95 <= aspectRatio <= 1.05:
            cv.drawContours(img_cv, [approx], 0, (0, 0, 255), 5)
            print(x, ',', y)
            cv.putText(img_cv, str((x, y)), (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
            cv.putText(img_cv, str((w, h)), (x + w, y + h), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
            cv.rectangle(img_cv, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), thickness=2)
            rect = cv.minAreaRect(approx)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(img_cv, [box], 0, (0, 0, 255), 2)
            colour = img_cv[int(y1 + h / 2)][int(x1 + w / 2)]

resized = cv.resize(img_cv, (900, 600), interpolation=cv.INTER_CUBIC)
cv.imshow('image_resized', resized)
# blank = np.zeros(img_cv.shape, np.uint8)
# cv.drawContours(blank, contours, -1, (0, 255, 0), thickness=2)
# cv.imshow('blank', blank)

cv.waitKey(0)
