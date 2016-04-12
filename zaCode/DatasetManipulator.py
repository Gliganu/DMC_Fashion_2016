import pandas as pd
import numpy as np
import  zaCode.FileManager as FileManager
from sklearn.cross_validation import train_test_split



def performDateEngineering(rawData, dateColumn):

    rawData[dateColumn+'-month']= rawData[dateColumn].map(lambda entryDate: float(entryDate.split("-")[1]))
    rawData[dateColumn+'-day'] = rawData[dateColumn].map(lambda entryDate: float(entryDate.split("-")[2]))

    rawData = rawData.drop([dateColumn], 1)

    return rawData


def performOHEOnColumn(data,columnName):

    #adding all the extra columns
    data = pd.concat([data, pd.get_dummies(data[columnName], prefix=columnName)], axis=1)

    #dropping the "source" column
    data = data.drop([columnName], 1)

    return data


def performSizeCodeEngineering(data):

    #drop everything that is not digit. About 200k examples ( maybe not the best way )
    data = data[data['sizeCode'].isdigit()]


    return data


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

    keptColumns = ['colorCode', 'quantity', 'price', 'rrp','deviceID','paymentMethod' ]

    if predictionColumnId:
        keptColumns.append(predictionColumnId)

    data = data[keptColumns]

    # drop NAs
    data = data.dropna()

    #restrict prediction to 0/1 for now
    filter = (data['returnQuantity'] == 0) | (data['returnQuantity'] == 1)
    data = data[filter]


    data = performOHEOnColumn(data, 'deviceID')

    data = performOHEOnColumn(data, 'paymentMethod')

    # data = performSizeCodeEngineering(data)

    # data = performDateEngineering(data, 'orderDate')

    print("Kept columns {}".format(data.columns))

    return data


def getTrainAndTestData():

    data = FileManager.get1000kTrainingData()

    predictionColumnId = 'returnQuantity'

    data = getFeatureEngineeredData(data,predictionColumnId)

    trainData, testData = train_test_split(data, test_size=0.25)

    xTrain = trainData.ix[:, trainData.columns != predictionColumnId].values
    yTrain = trainData[predictionColumnId].values

    xTest = testData.ix[:, testData.columns != predictionColumnId].values
    yTest = testData[predictionColumnId].values


    return xTrain,yTrain,xTest,yTest





















