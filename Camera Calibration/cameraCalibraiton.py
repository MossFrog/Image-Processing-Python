#-- Initial imports and required libraries --#
import numpy as np
import cv2
import glob

#-- Criteria to terminate the execution --#
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#-- Create an array of possible object points (Intuitive) --#
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

#-- Main array to store possible 3-D and 2-D points --#
threeDPoints = [] #-- 3D points in real world space. --#
twoDPoints = [] #-- 2D points on the image plane. --#

imagePath = 'distortedImage.jpg'

#-- Read the image from the hard drive --#
inputImage = cv2.imread(imagePath)
#-- Convert the image to grayscale --#
grayScale = cv2.cvtColor(inputImage,cv2.COLOR_BGR2GRAY)

#-- Detect the corners on the checkerboard utilizing openCV's built in method --#
ret, corners = cv2.findChessboardCorners(grayScale, (7,6),None)

#-- If the points are discovered then map them to 3-D points --#
if ret == True:
    threeDPoints.append(objp)

    corners2 = cv2.cornerSubPix(grayScale,corners,(11,11),(-1,-1),criteria)
    twoDPoints.append(corners2)

    #-- Render the corners and display them in a window named 'Output' --#
    inputImage = cv2.drawChessboardCorners(inputImage, (7,6), corners2,ret)
    cv2.imshow('Output',inputImage)

#-- Wait for 10 seconds or for an input --#
cv2.waitKey(10000)

#-- Calibration Section --#
#-- We retrieve the return values from he cv2.calibrateCamera function -- #
#-- These return values are variables such as the distortion coefficients, rotation and translation vectors --#
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(threeDPoints, twoDPoints, grayScale.shape[::-1],None,None)

#-- Get the height and width values of the input image --#
h,  w = inputImage.shape[:2]

#-- We have to ensure the last two elements of the distortion matrix are zero! --#
dist[0][3] = 0.0
dist[0][4] = 0.0

#-- Generate a new optimal camera matrix --#
newCamMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

#-- Apply the openCV undistort function --#
dst = cv2.undistort(inputImage, mtx, dist, None, newCamMatrix)

#-- Crop any excess from the image --#
x,y,w,h = roi
dst = dst[y:y + h, x:x + w]

#-- Display and write the undistorted image to the hard drive --#
cv2.imshow('Calibrated Result',dst)
cv2.imwrite('CalibratedResult.jpg', dst)
cv2.waitKey(10000)

cv2.destroyAllWindows()
