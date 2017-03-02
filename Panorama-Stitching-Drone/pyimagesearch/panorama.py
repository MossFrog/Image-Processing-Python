import numpy as np
import imutils
import cv2


def stitch(self, secondImage, firstImage, ratio=2.00, reprojectionThreshold=3.0):
	
	#-- After extracting the keypoints map them as descriptors --#
	(firstKeypoints, firstFeatures) = self.extractDescriptors(firstImage)
	(secondKeypoints, secondFeatures) = self.extractDescriptors(secondImage)

	#-- Detect the matching keypoints between the two images calling the keypointConnection method --#
	Matches = self.keypointConnection(firstKeypoints, secondKeypoints,
		firstFeatures, secondFeatures, ratio, reprojectionThreshold)

	#-- Using the homography calculated implement the cv2.warpPerspective method to stitch the two images togeather --#
	(resultingMatches, HomographyResult, status) = Matches

	retVal = cv2.warpPerspective(firstImage, HomographyResult,
		(firstImage.shape[1] + secondImage.shape[1], firstImage.shape[0]))
	retVal[0:secondImage.shape[0], 0:secondImage.shape[1]] = secondImage

	#-- Returning of the retVal as a raw image, can be displayed or saved --#
	return retVal



def keypointConnection(self, firstKeypoints, secondKeypoints, firstFeatures, secondFeatures,
	ratio, reprojectionThreshold):
	# compute the raw resultingMatches and initialize the list of actual
	# resultingMatches
	matchFinder = cv2.DescriptorMatcher_create("BruteForce")
	rawMatches = matchFinder.knnMatch(firstFeatures, secondFeatures, 2)
	resultingMatches = []

	# loop over the raw resultingMatches
	for m in rawMatches:
		# ensure the distance is within a certain ratio of each
		# other (i.e. Lowe's ratio test)
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			resultingMatches.append((m[0].trainIdx, m[0].queryIdx))

	# computing a homography requires at least 4 resultingMatches
	if len(resultingMatches) > 4:
		# construct the two sets of points
		firstPoints = np.float32([firstKeypoints[i] for (_, i) in resultingMatches])
		secondPoints = np.float32([secondKeypoints[i] for (i, _) in resultingMatches])

		# compute the homography between the two sets of points
		(HomographyResult, status) = cv2.findHomography(firstPoints, secondPoints, cv2.RANSAC,
			reprojectionThreshold)

		# return the resultingMatches along with the homograpy matrix
		# and status of each matched point
		return (resultingMatches, HomographyResult, status)

	# otherwise, no homograpy could be computed
	return None

