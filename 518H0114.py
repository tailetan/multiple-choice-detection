# import the necessary packages
import argparse
import cv2
import numpy as np
import math

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, convert it to grayscale 
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Convert to binary image
(thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv2.imshow('Bi ',im_bw)

# blur it slightly
blurred = cv2.GaussianBlur(im_bw, (5, 5), 0)


# apply Otsu's thresholding method to binarize the warped
# piece of paper
thresh = cv2.threshold(blurred, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cv2.imshow("thresh", thresh)

# Use morphological (opening)
kernelSize = (5, 5)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# cv2.imshow("Opening: ({}, {})".format(kernelSize[0], kernelSize[1]), opening)

#Compute edges
edges = cv2.Canny(opening, 50, 200)
# cv2.imshow("edges", edges)

cnts = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]

questionCnts = []
# loop over the contours
for c in cnts:
	# compute the bounding box of the contour, then use the
	# bounding box to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
 
	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
	if w >= 15 and h >= 15 and ar >= 0.75 and ar <= 1.18:
		questionCnts.append(c)

clone = image.copy()
cv2.drawContours(clone, questionCnts, -1, (0, 0, 255), 2)
cv2.imshow("Orginal image", image)
cv2.imshow("All Contours", clone)
cv2.waitKey(0)