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


def maximum(ar, wSize):
    max = 0
    for i in range(0, ar.shape[0] - wSize):
        for j in range(0, ar.shape[1] - wSize):
            for k in range(0, wSize):
                for l in range(0, wSize):
                  if(ar[i + k][j + l] > max):
                      max = ar[i + k][j + l]
                ar[i+wSize/2][j+wSize/2] = max
                max = 0
                
    return ar

#-- Common size method, scales the matrices to equal proportions
def common_size(a1, a2):
    (r1, c1) = a1.shape
    (r2, c2) = a2.shape
    return (a1[r1-r2 if r1>r2 else 0:,
               c1-c2 if c1>c2 else 0:],
            a2[r2-r1 if r2>r1 else 0::,
               c2-c1 if c2>c1 else 0:])

#-- Request the file name from the user

print("The image must be located within the directory of the Python Sript.")
fname = input('Enter the file name: ')
wSize = input('Enter the window size: ')

#-- Open and convert the image to a Numpy array
sourceImage = np_from_img(fname)

#-- Remove the image suffix from fname
fname = fname[:fname.find(".")]

#-- Uncomment to save the unedited Grayscale version
#save_as_img(norm(sourceImage), fname + "-Grayscale.png")

#-- Apply convolution and save
modImage = maximum(sourceImage, wSize)
save_as_img(sourceImage,
            fname + "-Maximum.png")

