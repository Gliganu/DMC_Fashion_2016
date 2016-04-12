import pandas as pd
import numpy as np
import  zaCode.FileManager as FileManager
from sklearn.cross_validation import train_test_split



def performDateEnginnering(rawData, dateColumn):

    rawData[dateColumn+'-month']= rawData[dateColumn].map(lambda entryDate: float(entryDate.split("-")[1]))
    rawData[dateColumn+'-day'] = rawData[dateColumn].map(lambda entryDate: float(entryDate.split("-")[2]))

    return rawData


def getFeatureEngineeredData(data,predictionColumnId = None):

    # orderID;
    # orderDate;
    # articleID;
    # colorCode;
    # sizeCode;
    # productGroup;
    # quantity;
    # price;
    # rrp;
    # voucherID;
    # voucherAmount;
    # customerID;
    # deviceID;
    # paymentMethod;
    # returnQuantity

    keptColumns = ['colorCode', 'quantity', 'price', 'rrp','deviceID' ]

    if predictionColumnId:
        keptColumns.append(predictionColumnId)

    print("Kept columns {}".format(keptColumns))

    data = data[keptColumns]

    # drop NAs
    data = data.dropna()

    #restrict prediction to 0/1 for now
    filter = (data['returnQuantity'] == 0) | (data['returnQuantity'] == 1)
    data = data[filter]


    # data = performDateEnginnering(data,'orderDate')
    # data = data.drop(['orderDate'],1)


    return data


def getTrainAndTestData():

    data = FileManager.getWholeTrainingData()

    predictionColumnId = 'returnQuantity'

    data = getFeatureEngineeredData(data,predictionColumnId)

    trainData, testData = train_test_split(data, test_size=0.25)

    xTrain = trainData.ix[:, trainData.columns != predictionColumnId].values
    yTrain = trainData[predictionColumnId].values

    xTest = testData.ix[:, testData.columns != predictionColumnId].values
    yTest = testData[predictionColumnId].values


    return xTrain,yTrain,xTest,yTest





















