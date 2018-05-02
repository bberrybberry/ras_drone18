import numpy as np
np.random.seed(1373) 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import cv2
import os
from os import listdir
from os.path import isfile, join
import scipy

batch_size = 128
nb_epoch = 50#12

nb_nodes = 64

nb_filters = 128

nb_pool = 2

nb_conv = 3


# Load letters data
imgHeight, imgWidth = 100, 100
downsampleFactor = 4
train_path = 'C:/Users/bbhig/Documents/Spring 2018/Computer Vision/final proj/graydata/train/'#'/mnt/c/Users/bbhig/Documents/Spring 2018/Computer Vision/final proj/graydata/train/'
test_path = 'C:/Users/bbhig/Documents/Spring 2018/Computer Vision/final proj/graydata/test/'#'/mnt/c/Users/bbhig/Documents/Spring 2018/Computer Vision/final proj/graydata/test/'

trainFilesList = [f for f in listdir(train_path) if isfile(join(train_path, f))]
testFilesList = [f for f in listdir(test_path) if isfile(join(test_path, f))]

numClasses = 26
numTrainSamples = len(trainFilesList)
numTestSamples = len(testFilesList)

trainData = np.zeros(
        (numTrainSamples,
         round(imgHeight/downsampleFactor), 
         round(imgWidth/downsampleFactor))) # numpy array of (num_samples, length * width)
testData = np.zeros(
        (numTestSamples, 
         round(imgHeight/downsampleFactor),
         round(imgWidth/downsampleFactor))) 

ind = 0
for file in trainFilesList:
    # read image data into training data
    img = cv2.imread('%s%s' % (train_path, file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    trainData[ind] = scipy.misc.imresize(gray, 1/downsampleFactor)
    ind = ind + 1

ind = 0
for file in testFilesList:
    img = cv2.imread('%s%s' % (test_path, file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    testData[ind] = scipy.misc.imresize(gray, 1/downsampleFactor)
    ind = ind + 1

# Generate labels
trainLabelsPerClass = numTrainSamples/numClasses
testLabelsPerClass = numTestSamples/numClasses
trainLabels = np.zeros(numTrainSamples)
testLabels = np.zeros(numTestSamples)
for n in range(0,26):
    for m in range(0, round(trainLabelsPerClass)):
        trainLabels[round(trainLabelsPerClass) * n + m] = n 
for n in range(0,26):
    for m in range(0, round(testLabelsPerClass)):
        testLabels[round(testLabelsPerClass) * n + m] = n 
    
X_train = trainData.reshape(trainData.shape[0], imgHeight//downsampleFactor, imgWidth//downsampleFactor,1)
X_test = testData.reshape(testData.shape[0], imgHeight//downsampleFactor, imgWidth//downsampleFactor,1)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


Y_train = np_utils.to_categorical(trainLabels, numClasses)
Y_test = np_utils.to_categorical(testLabels, numClasses)

model = Sequential()

model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid', input_shape=(imgHeight//downsampleFactor, imgWidth//downsampleFactor,1)))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='valid'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(nb_nodes))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(numClasses)) 
model.add(Activation('softmax')) 

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])

# Save models
save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'ocr.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test),callbacks=[checkpoint])

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

