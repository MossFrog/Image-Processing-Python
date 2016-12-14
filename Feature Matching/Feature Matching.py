import numpy as np
import cv2
from matplotlib import pyplot as plt

#-- Disable the usage of openCL for greater compatability --#
cv2.ocl.setUseOpenCL(False)

imageNo1 = cv2.imread('S03L.tif',0) #-- Image to be queried --#
imageNo2 = cv2.imread('S03R.tif',0) #-- Image to be matched --#

imageNo1 = imageNo1[200:600, 200:1000] #-- Crop out a certain Section of the Image --#

#-- Create the SIFT detector --#
orb = cv2.ORB_create()

#-- Detect and store key points within the images --#
keypoint1, descriptor1 = orb.detectAndCompute(imageNo1,None)
keypoint2, descriptor2 = orb.detectAndCompute(imageNo2,None)

#-- Initialize the BFMatcher object utilizing a Hamming Window --#
bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#-- Match the keypoints detected in descriptor1 and descriptor2 --#
matchResults = bfMatcher.match(descriptor1,descriptor2)

#-- Apply sorting to the matches according to their distances --#
matchResults = sorted(matchResults, key = lambda x:x.distance)

#-- Render the first 5 matches detected --#
resultImage = cv2.drawMatches(imageNo1,keypoint1,imageNo2,keypoint2,matchResults[:5], None, flags=2)

#-- Display the rendered image --#
plt.imshow(resultImage),plt.show()
