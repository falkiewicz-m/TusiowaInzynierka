#!/usr/bin/python
# @author	pctroll a.k.a Jorge Palacios 10-87970
# @file		iris_detection.py

import cv2
import math
import numpy as np
import os

# GLOBAL VARIABLES
#####################################
# Holds the pupil's center
centroid = (0,0)
# Holds the iris' radius
radius = 0
# Holds the current element of the image used by the getNewEye function
currentEye = 0
# Holds the list of eyes (filenames)
eyesList = []
#####################################


# Returns a different image filename on each call. If there are no more
# elements in the list of images, the function resets.
#
# @param list		List of images (filename)
# @return string	Next image (filename). Starts over when there are
#			no more elements in the list.
def getNewEye(list):
	global currentEye
	if (currentEye >= len(list)):
		currentEye = 0
	newEye = list[currentEye]
	currentEye += 1
	return (newEye)

# Returns the cropped image with the isolated iris and black-painted
# pupil. It uses the getCircles function in order to look for the best
# value for each image (eye) and then obtaining the iris radius in order
# to create the mask and crop.
#
# @param image		Image with black-painted pupil
# @returns image 	Image with isolated iris + black-painted pupil
def getIris(frame):
	iris = []
	copyImg = frame.copy()
	# cv2.imshow('copyImg', frame)
	resImg = frame.copy()
	grayImg = frame.copy()
	grayImg = cv2.cvtColor(grayImg, cv2.COLOR_BGR2GRAY)
	grayImg = cv2.Canny(grayImg, 5, 70)
	grayImg = cv2.GaussianBlur(grayImg, (7,7), 0)
	circles = getCircles(grayImg)
	print(circles)
	# cv2.imshow('c', circles)
	# cv2.imshow('resImg', resImg)
	iris.append(resImg)
	# for circle in circles:
	# 	rad = int(circle[2])
	# 	print(rad)
	# 	global radius
	# 	radius = rad
	# 	cv2.circle(grayImg, centroid, rad, (255,255,255), cv2.FILLED)
	# 	# cv2.Not(mask,mask)
	# 	# cv2.subtract(frame,copyImg,resImg,mask)
	# 	x = int(centroid[0] - rad)
	# 	y = int(centroid[1] - rad)
	# 	w = int(rad * 2)
	# 	h = w
	# 	resImg = frame[x:y, w:h]
	# 	cropImg = resImg.copy()
	# 	return(cropImg)
	# return (resImg)

# Search middle to big circles using the Hough Transform function
# and loop for testing values in the range [80,150]. When a circle is found,
# it returns a list with the circles' data structure. Otherwise, returns an empty list.

# @param image
# @returns list
def getCircles(image):
	i = 80
	while i < 151:
		# storage = cv.CreateMat(image.width, 1, cv.CV_32FC3)
		# storage = np.ndarray(shape=(image.shape[0], image.shape[1]))
		circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 2, 240.0)
		circles = np.round(circles[0,:]).astype("int")
		if (len(circles) == 1):
			return circles
		i +=1
	return ([])

# Returns the same images with the pupil masked black and set the global
# variable centroid according to calculations. It uses the FindContours
# function for finding the pupil, given a range of black tones.

# @param image		Original image for testing
# @returns image	Image with black-painted pupil
def getPupil(frame):

	cv2.imshow('in', frame)
	pupilImg = frame.copy()
	# cv2.inRange(frame, (240,240,240), (255,255,255), pupilImg)
	pupilImg = cv2.cvtColor(pupilImg, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(pupilImg, 80, 150, 0)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_KCOS)
	cv2.drawContours(thresh, contours, -1, (0,255,255), 2)
	cv2.imshow('threshold', thresh)
	cv2.imshow('out', pupilImg)
	# cnt = contours[0]
	# momnt = cv2.moments(cnt)
	# x = momnt['m10']/momnt['m00']
	# y = momnt['m01']/momnt['m00']
	del pupilImg
	pupilImg = frame.copy()
	while contours:
		moments = cv2.moments(im2)
		# print(moments)
		area = moments['m00']
		if (area > 50):
			pupilArea = area
			x = int(moments['m10']/area)
			y = int(moments['m01']/area)
			pupil = contours
			global centroid
			centroid = (int(x),int(y))
			cv2.circle(pupilImg, (x,y), 60, (0,255,0), 1)
			# cv2.drawContours(pupilImg, pupil, -1, (0,0,255), 1)
			cv2.imshow('petla', pupilImg)
			break
		contours = contours.h_next()
	return (pupilImg)

# Returns the image as a "tape" converting polar coord. to Cartesian coord.
#
# @param image		Image with iris and pupil
# @returns image	"Normalized" image
def getPolar2CartImg(image, rad):
	imgSize = cv.GetSize(image)
	c = (float(imgSize[0]/2.0), float(imgSize[1]/2.0))
	imgRes = cv.CreateImage((rad*3, int(360)), 8, 3)
	#cv.LogPolar(image,imgRes,c,50.0, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS)
	cv.LogPolar(image,imgRes,c,60.0, cv.CV_INTER_LINEAR+cv.CV_WARP_FILL_OUTLIERS)
	return (imgRes)

eyesList = os.listdir('images/eyes')
key = 0
while True:
	eye = getNewEye(eyesList)
	frame = cv2.imread("images/eyes/"+eye)
	iris = frame.copy()
	output = getPupil(frame)
	# iris = getIris(output)
	print(iris)
	# cv2.imshow('input',frame)
	# cv2.imshow('output',iris)
	normImg = iris.copy()
	# normImg = getPolar2CartImg(iris,radius)
	# cv.ShowImage("normalized", normImg)
	key = cv2.waitKey(24000) & 0xFF
	# seems like Esc with NumLck equals 1048603
	if (key == 27 or key == 1048603):
		break

cv2.DestroyAllWindows()
