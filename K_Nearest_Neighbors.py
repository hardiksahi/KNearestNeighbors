## HARDIK SAHI
## University of Waterloo

import numpy as np
import pandas as pd
import timeit
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import preprocessing


start = timeit.default_timer()

#10 fold cross validation
kFold = 10
kf = KFold(n_splits=kFold)

#Reading and normalizing the training and test data
training = pd.read_csv('MNIST_X_train.csv',header=None,nrows=60000).as_matrix()
trainOutput = pd.read_csv('MNIST_y_train.csv',header=None,nrows=60000).as_matrix()

training = preprocessing.normalize(training)

test = pd.read_csv('MNIST_X_test.csv',header=None,nrows=10000).as_matrix()
testOutput = pd.read_csv('MNIST_y_test.csv',header=None,nrows=10000).as_matrix()

test = preprocessing.normalize(test)

trainDotPrArray = np.zeros(training.shape[0]).astype(float)

for i in range(training.shape[0]):
    trainDotPrArray[i] = np.dot(training[i],training[i])


testDotPrArray = np.zeros(test.shape[0]).astype(float)
for i in range(test.shape[0]):
    testDotPrArray[i] = np.dot(test[i],test[i])


#Predict the class of the test point by getting the majority vote class of k nearest neighbors
def predictClass(indexArray, trainOutput):
    z = np.zeros((10,))
    for i in indexArray:
         z[trainOutput[i]]+=1
    return np.argmax(z)

#kNN algorithm : assigns one of the 10 possible classes to query point and returns corresponding error rate = ((errorCount*100)/# test data) 
def kNN(kVal,training,trainOutput,test,testOutput, trainDotPrArray,testDotPrArray,partitionN):
    errorCount = 0
    for i in range(test.shape[0]):
        testVector = test[i]
        trainTestDot = np.dot(training,testVector)# dot of a test data with entire training 
        testDotVector = np.full((training.shape[0],),testDotPrArray[i])
        distance =  trainDotPrArray + testDotVector - 2*trainTestDot # x.x + a.a - 2*a.x
        nearNeCount = kVal
        index_array = np.argpartition(distance, nearNeCount)[:nearNeCount] # Complexity O(nlog(n))
        predictedClass = predictClass(index_array,trainOutput)
        #print("Predicted otput class for test %d is %d and K %d partitionN %d"% (i+1,predictedClass, kVal,partitionN ))
        if predictedClass != testOutput[i]:
            errorCount+=1
    return (errorCount*100)/testOutput.shape[0]

kRange = [1,5,7,11,15,23,27,50,70] # Range of K values
kErrorArrayFinal = np.zeros((len(kRange),))
indexForFinalArray = 0

#10 fold cross validation
for k in (kRange):
    errorSum = 0.0
    partitionN = 0
    for train_index_tuple, valid_index_tuple in kf.split(training):
        partitionN+=1
        trainInputArray,validInputArray = training[train_index_tuple], training[valid_index_tuple]
        trainOutputArray,validOutputArray =trainOutput[train_index_tuple], trainOutput[valid_index_tuple]
        trainDotArray,validDotArray = trainDotPrArray[train_index_tuple], trainDotPrArray[valid_index_tuple]
        errorRate = kNN(k,trainInputArray, trainOutputArray, validInputArray,validOutputArray,trainDotArray,validDotArray,partitionN)
        errorSum+=errorRate
    meanCVErrorK = errorSum/kFold
    kErrorArrayFinal[indexForFinalArray] = meanCVErrorK
    indexForFinalArray+=1


# Using the best k from 10 fold cross validation to run on test data
leastErrorRateK = kRange[np.argmin(kErrorArrayFinal)]

#Error rate for test data
errorRateTest = kNN(leastErrorRateK, training,trainOutput, test, testOutput,trainDotPrArray, testDotPrArray,0)
print("Test error rate for best K %d is %f" % (leastErrorRateK,errorRateTest)) 

# Plot the graph of k versus Cross Validation error rate       
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.plot(kRange, kErrorArrayFinal, 'r-')
ax1.set_xlabel("K")
ax1.set_ylabel("CV error rate")

print("Error rate array",kErrorArrayFinal)

stop = timeit.default_timer()
print ("Time for code running ", stop - start)









    
    
