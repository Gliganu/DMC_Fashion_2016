import time
import numpy as np
import zaCode.ClassifierTrainer as ClassifierTrainer
import zaCode.DatasetManipulator as Toolbox
import zaCode.FileManager as FileManager
import zaCode.Validator as Validator
from zaCode import Visualizer
import pandas as pd
from sklearn.ensemble import VotingClassifier
from zaCode import TestPredictor


def addNewFeatures(data):
    data = Toolbox.constructOverpricedColumn(data)

    data = Toolbox.constructDiscountAmountColumn(data)

    data = Toolbox.constructOrderDuplicatesCountColumn(data)

    data = Toolbox.contructOrderDuplicatesDistinctColorColumn(data)

    data = Toolbox.constructOrderDuplicatesDistinctSizeColumn(data)

    data = Toolbox.constructArticleIdSuffixColumn(data)
    # data = data.drop(['articleID'],1)

    data = Toolbox.constructItemPercentageReturnColumn(data)

    data = Toolbox.constructBasketColumns(data)
    # data = data.drop(['orderID'], 1)

    return data


def engineerOldFeatures(data):
    data = Toolbox.performOHEOnColumn(data, 'deviceID', False)
    # data = data.drop(['deviceID'],1)

    data = Toolbox.performOHEOnColumn(data, 'paymentMethod', False)
    # data = data.drop(['paymentMethod'], 1)

    data = Toolbox.performSizeCodeEngineering(data)
    # data = data.drop(['sizeCode'], 1)

    # data = Toolbox.performColorCodeEngineering(data)

    data = Toolbox.performDateEngineering(data, 'orderDate')
    # data = data.drop(['orderDate'], 1)

    return data


def serializeDataFrame():
    fileName = 'allFeatures.csv'

    data = constructDataFromScratch()

    FileManager.saveDataFrame(data, fileName)


def constructDataFromScratch():
    data = FileManager.getRandomTrainingData(1000)
    # data = FileManager.get10kTrainingData()
    # data = FileManager.getWholeTrainingData()

    allDistinctOHEdata = FileManager.getDistinctOHEFeatures()
    data = pd.concat([data, allDistinctOHEdata], axis=0)

    keptColumns = ['orderDate', 'orderID', 'colorCode', 'quantity', 'price', 'rrp', 'deviceID', 'paymentMethod',
                   'productGroup',
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

    return data


def loadDataFrameFromCsv(fileName='allFeatures.csv', size=None):
    data = FileManager.loadDataFrameFromCsv(fileName, size=size)

    keptColumns = ['colorCode',
                   'quantity',
                   'price',
                   'rrp',
                   'normalisedSizeCode',
                   'voucherAmount',
                   'overpriced',
                   'discountedAmount',
                   'orderDuplicatesCount',
                   'articleIdSuffix',
                   'itemPercentageReturned',
                   'basketSize',
                   'basketQuantity',
                   'deviceID_1', 'deviceID_2', 'deviceID_3', 'deviceID_4', 'deviceID_5',
                   'paymentMethod_BPLS', 'paymentMethod_BPPL', 'paymentMethod_BPRG', 'paymentMethod_CBA',
                   'paymentMethod_KGRG', 'paymentMethod_KKE', 'paymentMethod_NN', 'paymentMethod_PAYPALVC',
                   'paymentMethod_RG', 'paymentMethod_VORAUS',
                   'weekday',
                   'orderDate-month',
                   'orderDate-day',
                   'returnQuantity',
                   'productGroup',
                   'deviceID',
                   'paymentMethod',
                   'customerID',
                   ]

    data = Toolbox.filterColumns(data, keptColumns)
    return data


def makePrediction():
    print("Reading data...")

    # data = constructDataFromScratch()
    data = loadDataFrameFromCsv(size=100000)

    # Construct polynomial features
    # polynomialFeaturesSourceColumns = ['quantity', 'price', 'voucherAmount', 'basketQuantity', 'itemPercentageReturned', 'overpriced', 'discountedAmount']

    polynomialFeaturesSourceColumns = ['colorCode',
                                       'quantity',
                                       'price',
                                       'rrp',
                                       'normalisedSizeCode',
                                       'voucherAmount',
                                       'overpriced',
                                       'discountedAmount',
                                       'orderDuplicatesCount',
                                       'articleIdSuffix',
                                       'itemPercentageReturned',
                                       'basketSize',
                                       'basketQuantity'
                                       ]

    # data = Toolbox.constructPolynomialFeatures(data,polynomialFeaturesSourceColumns,degree=2, interaction_only=False)

    # Split into train/test data
    trainData, testData = Toolbox.performTrainTestSplit(data, 0.25)

    # construct median color/size per customer + difference
    trainData, testData = Toolbox.constructCustomerMedianSizeAndColor(trainData, testData)

    # construct the percentage return column
    trainData, _, _ = Toolbox.constructPercentageReturnColumnForSeenCustomers(trainData, testData)

    trainData = trainData.drop(['productGroup', 'deviceID', 'paymentMethod'], 1)
    testData = testData.drop(['productGroup', 'deviceID', 'paymentMethod'], 1)

    trainData = trainData.drop(['customerID'], 1)
    testData = testData.drop(['customerID'], 1)

    print("\n\nFinal columns {} : {}".format(len(trainData.columns), trainData.columns))

    # X and Y train
    xTrain, yTrain = Toolbox.getXandYMatrix(trainData, 'returnQuantity')

    # Select K best features according to variance
    # xTrain, selectedColumns = Toolbox.selectKBest(xTrain, yTrain, 65, trainData.columns)
    # testData = testData[selectedColumns].copy()

    # X and Y test
    xTest, yTest = Toolbox.getXandYMatrix(testData, 'returnQuantity')

    # apply PCA
    # xTrain,xTest = Toolbox.performPCA(xTrain,xTest,65)

    # apply RMB Transform ( + normalize and binarize )
    # xTrain,xTest = Toolbox.performRBMTransform(xTrain,xTest)

    # Training the classifier
    classifier = ClassifierTrainer.trainClassifier(xTrain, yTrain)
    yPred = classifier.predict(xTest)

    # yPred = Toolbox.predictUsingVotingClassifier(xTest)

    # FileManager.saveModel(classifier,'withPercentage/gradientBoosting', 'gradientBoosting.pkl')


    # Assessing the performance
    Validator.performValidation(yPred, yTest)
    Visualizer.plotFeatureImportance(classifier.feature_importances_,
                                     [col for col in trainData.columns if col != 'returnQuantity'])


def makePredictionUsingDoubleClassifiers():
    print("Reading data...")

    # data = constructDataFromScratch()
    data = loadDataFrameFromCsv(size=100000)

    # Split into train/test data
    trainData, testData = Toolbox.performTrainTestSplit(data, 0.25)

    # construct median color/size per customer + difference
    trainData, testData = Toolbox.constructCustomerMedianSizeAndColor(trainData, testData)

    # construct the percentage return column
    trainDataExtra, testData, customerIDToPercReturnedDict = Toolbox.constructPercentageReturnColumnForSeenCustomers(
        trainData, testData)

    trainDataExtra = trainDataExtra.drop(['productGroup', 'deviceID', 'paymentMethod', 'customerID'], 1)
    trainData = trainData.drop(['productGroup', 'deviceID', 'paymentMethod', 'customerID'], 1)
    testData = testData.drop(['productGroup', 'deviceID', 'paymentMethod'], 1)

    print("\n\nFinal columns {} : {}".format(len(trainData.columns), trainData.columns))

    # X and Y train
    xTrain, yTrain = Toolbox.getXandYMatrix(trainData, 'returnQuantity')
    xTrainExtra, _ = Toolbox.getXandYMatrix(trainDataExtra, 'returnQuantity')
    _, yTest = Toolbox.getXandYMatrix(testData, 'returnQuantity')

    # Training the classifier
    classifier = ClassifierTrainer.trainClassifier(xTrain, yTrain)
    classifierExtra = ClassifierTrainer.trainClassifier(xTrainExtra, yTrain)

    # todo SILVI
    yPred = TestPredictor.makePrediction(classifier, classifierExtra, testData, customerIDToPercReturnedDict,
                                         ['customerID', 'percentageReturned', 'returnQuantity'],
                                         ['customerID', 'returnQuantity'])

    # yPred = classifier.predict(xTest)
    # yPred = Toolbox.predictUsingVotingClassifier(xTest)

    # FileManager.saveModel(classifier,'withPercentage/gradientBoosting', 'gradientBoosting.pkl')

    # Assessing the performance
    Validator.performValidation(yPred, yTest)
    Visualizer.plotFeatureImportance(classifier.feature_importances_,
                                     [col for col in trainData.columns if col != 'returnQuantity'])


if __name__ == '__main__':
    startTime = time.time()

    # makePrediction()
    makePredictionUsingDoubleClassifiers()
    # serializeDataFrame()

    # Visualizer.calculateLearningCurve()
    # Visualizer.calculateRocCurve()


    endTime = time.time()
    print("\nTotal run time:{}".format(endTime - startTime))
