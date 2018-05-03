import numpy as np
np.random.seed(1373) 
from keras.models import load_model
import cv2
import os
from os import listdir
from os.path import isfile, join

theAlphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# load model
model = load_model('C:/Projects/dronekit-python/ras_drone18/models/ocrV2.020.h5')
#model = load_model('C:/Projects/dronekit-python/ras_drone18/ocr.033.h5')
print("Loading OCR V2")

# read some data
# Load letters data

imgHeight, imgWidth = 25, 25
numSamples = 19
path = 'C:/Projects/dronekit-python/ras_drone18/img_data/simple_suas_cropped/'
filesList = [f for f in listdir(path) if isfile(join(path, f))]
filename = 'A_002.PNG'

evalData = np.zeros(
        (numSamples,
         round(imgHeight), 
         round(imgWidth), 1)) # numpy array of (num_samples, length * width)

ind = 0
for file in filesList:
    # read image data into training data
    img = cv2.imread('%s%s' % (path, file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
    gray = clahe.apply(gray)
    imgIn = cv2.resize(gray, (imgHeight, imgWidth))
    imgIn = imgIn.reshape(1, imgHeight, imgWidth,1)
    imgIn = imgIn.astype('float32')
    imgIn /= 255
    evalData[ind] = imgIn
    ind = ind + 1
    #cv2.imshow('Image', imgIn)
    #cv2.waitKey(0)



#path = 'C:/Users/bbhig/Documents/Spring 2018/Computer Vision/final proj/graydata/test/'
#filename = 'G_75.png'
#img = cv2.imread('%s%s' % (path, filename))
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##gray = cv2.equalizeHist(gray)
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
#gray = clahe.apply(gray)
#imgIn = cv2.resize(gray, (imgHeight, imgWidth))

# predict on model

for n in range(0, numSamples):
    filename = filesList[n]
    img = evalData[n]
    cv2.imwrite('input_'+filename[:-4]+'.PNG', img.reshape(imgHeight, imgWidth))
    img = img.reshape(1, imgHeight, imgWidth,1)
    print("Reading file: %s" % (filename))
    ans = model.predict(img, verbose = 1, steps=1)
    print('Predicted: %s' % theAlphabet[np.argmax(ans)])