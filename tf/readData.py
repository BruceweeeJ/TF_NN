import os
import numpy as np
def read():
    f = open('D:/MyWorkSpace/GAN&Tensorflow/data/MNIST/train-images.idx3-ubyte')
    load =np.fromfile(file=f,dtype=np.uint8)
    trainX = load[16:].reshape((60000,784)).astype(np.float)
    f = open('D:/MyWorkSpace/GAN&Tensorflow/data/MNIST/train-labels.idx1-ubyte')
    load = np.fromfile(file=f, dtype=np.uint8)
    trainY = load[8:].reshape((60000)).astype(np.float)
    f = open('D:/MyWorkSpace/GAN&Tensorflow/data/MNIST/t10k-images.idx3-ubyte')
    load = np.fromfile(file=f, dtype=np.uint8)
    testX = load[16:].reshape((10000,  784)).astype(np.float)
    f = open('D:/MyWorkSpace/GAN&Tensorflow/data/MNIST/t10k-labels.idx1-ubyte')
    load = np.fromfile(file=f, dtype=np.uint8)
    testY = load[8:].reshape((10000)).astype(np.float)
    trainY = np.asarray(trainY)
    testY = np.asarray(testY)
    
    np.random.seed(547)
    np.random.shuffle(trainX)
    np.random.seed(547)
    np.random.shuffle(trainY)
    np.random.seed(547)
    np.random.shuffle(testX)
    np.random.seed(547)
    np.random.shuffle(testY)

    y_train_vec = np.zeros((len(trainY), 10), dtype=np.float)
    y_test_vec = np.zeros((len(testY), 10), dtype=np.float)
    trainY = trainY.astype(np.int)
    for i in range(len(trainY)):
        y_train_vec[i][trainY[i]]=1

    testY = testY.astype(np.int)
    for i in range(len(testY)):
        y_test_vec[i][testY[i]] = 1

    return trainX/255,y_train_vec,testX/255,y_test_vec
def printimg(onepic):
    onepic=onepic.reshape(28,28)
    for i in range(28):
        for j in range(28):
            if onepic[i,j]==0:
                print('  ',end='')
            else:
                print('* ',end='')
        print('')
trainX,trainY,testX,testY=read()
for i in trainX[:100]:
    printimg(i)