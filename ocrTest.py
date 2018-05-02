import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Load letters data
imgHeight, imgWidth = 100, 100
train_path = '/mnt/c/Users/bbhig/Documents/Spring 2018/Computer Vision/final proj/graydata/train/'
test_path = '/mnt/c/Users/bbhig/Documents/Spring 2018/Computer Vision/final proj/graydata/test/'

trainFilesList = [f for f in listdir(train_path) if isfile(join(train_path, f))]
testFilesList = [f for f in listdir(test_path) if isfile(join(test_path, f))]

numClasses = 26
numTrainSamples = len(trainFilesList)
numTestSamples = len(testFilesList)

trainData = [] # numpy array of (num_samples, length * width)
testData = []

for file in listdir(train_path):
	if isfile(file):
		# read image data into training data
		img = cv2.imread('%s%s' % (train_path, file))
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		# flatten px intensity array
		data = gray.reshape(-1, imgHeight * imgWidth).astype(np.float32)
		trainData.append(data)

for file in listdir(test_path):
	if isfile(file):
		# read image data into training data
		img = cv2.imread('%s%s' % (test_path, file))
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		# flatten px intensity array
		data = gray.reshape(-1, imgHeight * imgWidth).astype(np.float32)
		testData.append(data)

SZ=20
bin_n = 16 # Number of bins


svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
                    svm_type = cv2.ml.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

# def deskew(img):
    # m = cv2.moments(img)
    # if abs(m['mu02']) < 1e-2:
        # return img.copy()
    # skew = m['mu11']/m['mu02']
    # M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    # return img

# def hog(img):
    # gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    # gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    # mag, ang = cv2.cartToPolar(gx, gy)
    # bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    # bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    # mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    # hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    # hist = np.hstack(hists)     # hist is a 64 bit vector
    # return hist

# img = cv2.imread('digits.png',0)

# cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

# First half is trainData, remaining is testData
# train_cells = [ i[:50] for i in cells ]
# test_cells = [ i[50:] for i in cells]

######     Now training      ########################

# deskewed = [map(deskew,row) for row in train_cells]
# hogdata = [map(hog,row) for row in deskewed]
# trainData = np.float32(hogdata).reshape(-1,64)
responses = np.float32(np.repeat(np.arange(numClasses),numTestSamples)[:,np.newaxis])

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
svm.train(trainingDataMat, cv2.ml.ROW_SAMPLE, labelsMat)

sample_data = np.array([*your_data*], np.float32)
response = svm.predict(sample_data)


SVM = cv2.ml.SVM_create()
SVM.setKernel(cv2.ml.SVM_LINEAR)
SVM.setP(0.2)
SVM.setType(cv2.ml.SVM_EPS_SVR)
SVM.setC(1.0)

#training
SVM.train_auto(samples, cv2.ml.ROW_SAMPLE, responses)

#predict
output = SVM.predict(samples)[1].ravel()

svm.train(trainData,responses, params=svm_params)
svm.save('svm_data.dat')

######     Now testing      ########################

# deskewed = [map(deskew,row) for row in test_cells]
# hogdata = [map(hog,row) for row in deskewed]
# testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict_all(testData)

#######   Check Accuracy   ########################
mask = result==responses
correct = np.count_nonzero(mask)
print correct*100.0/result.size