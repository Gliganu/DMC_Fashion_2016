import pandas as pd
import numpy as np
import math
import sys

from datetime import datetime
from collections import defaultdict
from copy import copy

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler, PolynomialFeatures, normalize
from sklearn.feature_selection import SelectKBest, f_regression

import zaCode.FileManager as FileManager
import zaCode.DatasetManipulator as Toolbox
import zaCode.Validator as Validator
import zaCode.ClassifierTrainer as ClassifierTrainer
import time


def addNewFeatures(data):

    data = Toolbox.constructOverpricedColumn(data)

    data = Toolbox.constructDiscountAmountColumn(data)

    data = Toolbox.constructPercentageReturnColumn(data)

    data = Toolbox.constructItemPercentageReturnColumn(data)

    data = Toolbox.constructBasketColumns(data)

    return data


def engineerOldFeatures(data):
    data = Toolbox.performOHEOnColumn(data, 'deviceID')

    data = Toolbox.performOHEOnColumn(data, 'paymentMethod')

    data = Toolbox.performSizeCodeEngineering(data)

    data = Toolbox.performColorCodeEngineering(data)

    data = Toolbox.performDateEngineering(data, 'orderDate')

    return data

def makePrediction():
    print("Reading data...")
    data = FileManager.get100kTrainingData()

    keptColumns = ['orderDate', 'orderID', 'colorCode', 'quantity', 'price', 'rrp', 'deviceID', 'paymentMethod',
                   'sizeCode', 'voucherAmount', 'customerID', 'articleID', 'returnQuantity']

    # Keep just the columns of interest
    data = Toolbox.filterColumns(data, keptColumns)

    # Restrict prediction to 0/1 for now. Map everything greater than 1 to 1
    data = Toolbox.restrictReturnQuantityToBinaryChoice(data)

    # Handle missing values
    data = Toolbox.dropMissingValues(data)

    # Construct additional features
    data = addNewFeatures(data)

    # Perform feature engineering on existing features
    data = engineerOldFeatures(data)

    # Construct polynomial features
    # polynomialFeaturesSourceColumns = ['quantity', 'price', 'voucherAmount', 'basketQuantity', 'percentageReturned', 'overpriced',
    #             'discountedAmount']
    polynomialFeaturesSourceColumns = data.columns
    data = Toolbox.constructPolynomialFeatures(data, polynomialFeaturesSourceColumns,degree=2, interaction_only=False)


    #Split into train/test data
    trainData, testData = Toolbox.performTrainTestSplit(data,0.25)

    #X and Y train
    xTrain,yTrain = Toolbox.getXandYMatrix(trainData,'returnQuantity')

    #Select K best features according to variance
    xTrain, selectedColumns = Toolbox.selectKBest(xTrain, yTrain, 40, data.columns)
    testData = testData[selectedColumns].copy()

    # X and Y test
    xTest, yTest = Toolbox.getXandYMatrix(testData, 'returnQuantity')

    #Training the classifier
    classifier = ClassifierTrainer.trainClassifier(xTrain, yTrain)

    #Predicting
    yPred = classifier.predict(xTest)

    #Assessing the performance
    Validator.performValidation(yPred, yTest)



if __name__ == '__main__':
    startTime = time.time()

    makePrediction()

    # Visualizer.calculateLearningCurve(keptColumns)
    # Visualizer.calculateRocCurve()


    endTime = time.time()
    print("\nTotal run time:{}".format(endTime - startTime))