from scipy import signal as sg
from PIL import Image
import numpy as np
import struct

#-- Method for extracting RGB from Hexadecimal input
def hexToRGB(inStr):
    return struct.unpack('BBB',inStr.decode('hex'))

#-- Definitions from 'http://juanreyero.com/article/python/python-convolution.html'
#-- Modified for grayscale conversion.
def np_from_img(fname):
    return np.asarray(Image.open(fname).convert('L'), dtype=np.float32)
def np_rgba_from_img(fname):
    return np.asarray(Image.open(fname).convert('RGBA'), dtype=np.float32)

def save_as_img(ar, fname):
    Image.fromarray(ar.round().astype(np.uint8)).save(fname)

def norm(ar):
    return 255.*np.absolute(ar)/np.max(ar)


#-- Custom binary image segmentation method
def color(ar, C1, C2):
    
    for i in range(0, ar.shape[0]):
        for j in range(0, ar.shape[1]):
            if(ar[i][j][0] > 100.):
                ar[i][j][0] = hexToRGB(C1)[0]
                ar[i][j][1] = hexToRGB(C1)[1]
                ar[i][j][2] = hexToRGB(C1)[2]
                ar[i][j][3] = 255
            else:
                ar[i][j][0] = hexToRGB(C2)[0]
                ar[i][j][1] = hexToRGB(C2)[1]
                ar[i][j][2] = hexToRGB(C2)[2]
                ar[i][j][3] = 255
                
    return ar

#-- Common size method, scales the matrices to equal proportions
def common_size(a1, a2):
    (r1, c1) = a1.shape
    (r2, c2) = a2.shape
    return (a1[r1-r2 if r1>r2 else 0:,
               c1-c2 if c1>c2 else 0:],
            a2[r2-r1 if r2>r1 else 0::,
               c2-c1 if c2>c1 else 0:])

#-- Convolution method for edge detection.
def edgeDetect(im):
    imv, imh = common_size(sg.convolve(im, [[1., -1.]]),
                           sg.convolve(im, [[1.], [-1.]]))
    return np.sqrt(np.power(imv, 2)+np.power(imh, 2))

#-- Request the file name from the user

print("The image must be located within the directory of the Python Sript.")
fname = input('Enter the file name: ')

#-- A few simpe kernels

kernel_sharpen = np.array([[0.,-1.,0.],
                           [-1.,5.,-1.],
                           [0.,-1.,0.]])

kernel_blur = np.array([[1.,1.,1.],
                        [1.,1.,1.],
                        [1.,1.,1.]])


kernel_multi_edge = np.array([[1.,0.,-1.],
                              [0.,0.,0.],
                              [-1.,0.,1.]])

kernel_dilate = np.array([[0.,1.,0.],
                          [1.,1.,1.],
                          [0.,1.,0.]])


#-- Gaussian kernel calculated @ 'http://dev.theomader.com/gaussian-kernel-calculator/'
kernel_gauss = np.array([[0.102059,0.115349,0.102059],
                        [0.115349,0.130371,0.115349],
                        [0.102059,0.115349,0.102059]])

#-- Open and convert the image to a Numpy array
sourceImage = np_from_img(fname)

#-- Remove the image suffix from fname
fname = fname[:fname.find(".")]

#-- Save the unedited Grayscale version
save_as_img(norm(sourceImage), fname + "-Grayscale.png")

#-- Apply and save convolved images
save_as_img(norm(sg.convolve(sourceImage, [[1.],[-1.]])),
            fname + "-H.png")

save_as_img(norm(sg.convolve(sourceImage, [[1., -1.]])),
            fname + "-V.png")

edgeMatrix = edgeDetect(sourceImage)

save_as_img(norm(edgeMatrix), fname + "-Edge.png")

blurredMatrix = norm(sg.convolve(sourceImage, kernel_blur))

save_as_img(blurredMatrix,
            fname + "-Blurred.png")

#-- Reload the blurredMatrix and apply a crop to remove the border pixels.
blurredMatrix = Image.open(fname + "-Blurred.png").convert('L')
blurredMatrix = blurredMatrix.crop((1, 1, blurredMatrix.width-1, blurredMatrix.height-1))
blurredMatrix = np.asarray(blurredMatrix, dtype=np.float32)

#-- Subtract the blurred matrix from the source image.
sharpMatrix = np.subtract(blurredMatrix, sourceImage)

save_as_img(norm(sharpMatrix),
            fname + "Sub-Sharper.png")

save_as_img(norm(sg.convolve(sourceImage, kernel_sharpen)),
            fname + "-Sharper.png")

save_as_img(norm(sg.convolve(sourceImage, kernel_gauss)),
            fname + "-Gaussian.png")

#-- Apply the Erosion+Dilation methods

#-- Erode four times consecutively
resultMatrix = sg.convolve(sourceImage, kernel_blur)
resultMatrix = sg.convolve(resultMatrix, kernel_blur)
resultMatrix = sg.convolve(resultMatrix, kernel_blur)
resultMatrix = sg.convolve(resultMatrix, kernel_blur)


#-- Dilate the matrix four times
resultMatrix = sg.convolve(resultMatrix, kernel_dilate)
resultMatrix = sg.convolve(resultMatrix, kernel_dilate)
resultMatrix = sg.convolve(resultMatrix, kernel_dilate)
resultMatrix = sg.convolve(resultMatrix, kernel_dilate)

save_as_img(norm(resultMatrix),
            fname + "-Cleaned.png")

#-- Request user input for color thresholds
C1 = input('Enter Colour No.1 For Threshold (HEX): ')
C2 = input('Enter Colour No.2 For Threshold (HEX): ')

#-- Reload the Cleaned image
modImage = np_rgba_from_img(fname + "-Cleaned.png")
modImage = color(modImage, C1, C2)
save_as_img(modImage,
            fname + "-Segmented.png")



#-- Display ending notification
print("--- Operation Complete ---")
