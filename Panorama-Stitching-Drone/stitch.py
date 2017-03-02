# USAGE
# python stitch.py --first images/sharpenedFrame74.jpg --second images/sharpenedFrame75.jpg

# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
args = vars(ap.parse_args())

imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
imageA = imutils.resize(imageA, width=500)
imageB = imutils.resize(imageB, width=500)

# stitch the images together to create a panorama
stitcher = Stitcher()
(result) = stitcher.stitch([imageA, imageB])

# show the images
#cv2.imshow("Image A", imageA)
#cv2.imshow("Image B", imageB)
cv2.imshow("Result", result)
cv2.imshow("Stacked", np.hstack((imageA,imageB)))
cv2.imwrite("Result.jpg", result)
cv2.waitKey(0)