import pandas as pd
import numpy as np
import  zaCode.FileManager as FileManager
from sklearn.cross_validation import train_test_split
import math as math
from sklearn.preprocessing import Imputer


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
    data = data[data['sizeCode'].apply(lambda x: x.isnumeric())]

    return data


def constructPercentageReturnColumn(data):


    dataByCustomer = data[['quantity', 'returnQuantity']].groupby(data['customerID'])

    dataSummedByCustomer = dataByCustomer.apply(sum)
    dataSummedByCustomer['percentageReturned'] = dataSummedByCustomer['returnQuantity'] / dataSummedByCustomer['quantity']
    dataSummedByCustomer = dataSummedByCustomer.drop(['returnQuantity', 'quantity'], 1)

    idToPercDict = dataSummedByCustomer.to_dict().get('percentageReturned')

    data['percentageReturned'] = data['customerID'].apply(lambda custId: idToPercDict[custId])

    data = data.drop(['customerID'], 1)

    return data

def addNewFeatures(data):

    #see whether the product was overpriced. price > recommended
    data['overpriced'] = data['price'] > data['rrp']

    #see how much the data was discounted ( if price == 0, divide by 1 )
    data['discountedAmount'] = data['voucherAmount'] / data['price'].apply(lambda pr: max(pr,1))

    data =constructPercentageReturnColumn(data)
    return data


def performColorCodeEngineering(data):

    #get the thousands digit. in the color RAL system, that represents the "Base" color
    data['colorCode'] = data['colorCode'].apply(lambda code: code/1000)

    return data


def handleMissingValues(data):

    #dropping the rows with NA values
    data = data.dropna()



    return data

def getFeatureEngineeredData(data,predictionColumnId = None):

    print ("Performing feature engineering...")
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

    keptColumns = ['colorCode', 'quantity', 'price', 'rrp','deviceID','paymentMethod','sizeCode','voucherAmount','customerID' ]

    if predictionColumnId:
        keptColumns.append(predictionColumnId)

    data = data[keptColumns]

    # construct additional features as a mixture of various ones
    data = addNewFeatures(data)

    # drop NAs
    data = handleMissingValues(data)


    #restrict prediction to 0/1 for now. Map everything greater than 1 to 1
    data['returnQuantity'] = data['returnQuantity'].apply(lambda retQuant: min(retQuant,1))


    data = performOHEOnColumn(data, 'deviceID')

    data = performOHEOnColumn(data, 'paymentMethod')

    data = performSizeCodeEngineering(data)

    data = performColorCodeEngineering(data)

    # data = performDateEngineering(data, 'orderDate')

    print("\nKept columns {}".format(data.columns))

    return data


def getTrainAndTestData():

    print("Reading CSV...")
    data = FileManager.get1000kTrainingData()

    predictionColumnId = 'returnQuantity'

    data = getFeatureEngineeredData(data,predictionColumnId)

    trainData, testData = train_test_split(data, test_size=0.25)

    xTrain = trainData.ix[:, trainData.columns != predictionColumnId].values
    yTrain = trainData[predictionColumnId].values

    xTest = testData.ix[:, testData.columns != predictionColumnId].values
    yTest = testData[predictionColumnId].values


    return xTrain,yTrain,xTest,yTest





















