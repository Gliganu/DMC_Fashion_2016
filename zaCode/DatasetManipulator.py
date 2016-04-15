import pandas as pd
import numpy as np
import FileManager
from sklearn.cross_validation import train_test_split



def normalizeSize(data):
    data['sizeCode'][data['sizeCode'] == 'XS'] = 0
    data['sizeCode'][data['sizeCode'] == 'S'] = 1
    data['sizeCode'][data['sizeCode'] == 'M'] = 2
    data['sizeCode'][data['sizeCode'] == 'L'] = 3
    data['sizeCode'][data['sizeCode'] == 'XL'] = 4
    # set A and I on 2 for now
    data['sizeCode'][data['sizeCode'] == 'I'] = 2
    data['sizeCode'][data['sizeCode'] == 'A'] = 2
    data['sizeCode'] = pd.to_numeric(data['sizeCode'])
    # normalize XS-XL
    data['sizeCode'][(data['sizeCode'] <= 4) & (data['sizeCode'] >= 0)] = \
    data['sizeCode'][(data['sizeCode'] <= 4) & (data['sizeCode'] >= 0)].apply(lambda x: x / 4.0)
    # normalize 32-44
    data['sizeCode'][(data['sizeCode'] <= 44) & (data['sizeCode'] >= 32)] = \
    data['sizeCode'][(data['sizeCode'] <= 44) & (data['sizeCode'] >= 32)].apply(lambda x:(x - 32.0) / (44.0 - 32.0))
    # normalize 75-100
    data['sizeCode'][(data['sizeCode'] <= 100) & (data['sizeCode'] >= 75)] = \
    data['sizeCode'][(data['sizeCode'] <= 100) & (data['sizeCode'] >= 75)].apply(lambda x:(x - 75.0) / (100.0 - 75.0))
    # set I and A to mean
    data['sizeCode'][data['sizeCode'] == 2] = data['sizeCode'].mean()



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

    keptColumns = ['colorCode', 'quantity', 'price', 'rrp']

    if predictionColumnId:
        keptColumns.append(predictionColumnId)


    print "Kept columns {}".format(keptColumns)

    data = data[keptColumns]

    # drop NAs
    data = data.dropna()

    #restrict prediction to 0/1 for now
    filter = (data['returnQuantity'] == 0) | (data['returnQuantity'] == 1)
    data = data[filter]

    return data


# if return quantity is larger than zero, set it to 0
def getFeatureEngineeredDataTresholded(data,predictionColumnId = None):

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

    keptColumns = ['productGroup', 'colorCode', 'quantity', 'price', 'deviceID', 'rrp']

    if predictionColumnId:
        keptColumns.append(predictionColumnId)


    print "Kept columns {}".format(keptColumns)

    data = data[keptColumns]

    # drop NAs
    data = data.dropna()

    #restrict prediction to 0/1 for now
    data['returnQuantity'][data['returnQuantity'] > 0] = 1

    return data


def getTrainAndTestData():

    data = FileManager.getWholeTrainingData()

    predictionColumnId = 'returnQuantity'

    data = getFeatureEngineeredDataTresholded(data,predictionColumnId)

    trainData, testData = train_test_split(data, test_size=0.25)

    xTrain = trainData.ix[:, trainData.columns != predictionColumnId].values
    yTrain = trainData[predictionColumnId].values

    xTest = testData.ix[:, testData.columns != predictionColumnId].values
    yTest = testData[predictionColumnId].values


    return xTrain,yTrain,xTest,yTest





















