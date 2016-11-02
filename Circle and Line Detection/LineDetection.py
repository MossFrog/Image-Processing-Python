import cv2
import numpy as np

img = cv2.imread('sudoku.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imwrite('IntermediateOutput.jpg',edges)

minLineLength = 999999
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,150,minLineLength,maxLineGap)


for i in range(len(lines)):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)

cv2.imwrite('HoughLineOutput.jpg',img)
