import pandas as pd
import numpy as np
import math
import sys

from collections import defaultdict

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer

import  zaCode.FileManager as FileManager


def mapFeaturesToCondProbs(rawData, featureMask = None):
    """
        replaces all features in the dataset with the 
        conditional probability of the return qty being 
        equal to or greater than one i.e. maps:
        feature -> P(returnQuantity > 0 | Feature = feature).
        
        param featureMask : dict of colName -> bool that 
                            controls which feature is mapped
                            default = None, will map all
        return modified data and global mapping used in tuple
    """
    if featureMask == None:
        featureMask = defaultdict(lambda: True, returnQuantity=False)
    
    rowNum = len(rawData)
    # will also return a global map so we can do the inverse mappings
    globalMap = {}
    
    for colName, series in rawData.iteritems():

        if featureMask[colName]:
            print('processing column ' + colName + '...')
            colMap = {}
            for uniqueVal in series.unique():
                allValsCnt = series[series == uniqueVal].count();
                valsFavorableSeries = (rawData[rawData['returnQuantity'] > 0])[colName]
                valsFavorableCnt = valsFavorableSeries[valsFavorableSeries == uniqueVal].count()
                
                colMap[uniqueVal] = valsFavorableCnt / allValsCnt
                
            globalMap[colName] = colMap
            
            # actually apply transform
            series = series.apply(lambda x: colMap[x])
            rawData[colName] = series
            
    return rawData, globalMap       
    
def clusterRetQtyGTOne(rawData):
    """
        replaces all return qts greater than one
        with one, turning problem into binary classification.
    """
    rawData['returnQuantity'] = rawData['returnQuantity'].apply(lambda qty: 0 if qty < 1 else 1)
    
    return rawData

        
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
    
    # avoind chain indexing warning
    return data.copy()


def constructPercentageReturnColumn(data):
    # avoid chain indexing warning
    dataCopy = data.copy()

    dataByCustomer = dataCopy[['quantity', 'returnQuantity']].groupby(dataCopy['customerID'])

    dataSummedByCustomer = dataByCustomer.apply(sum)
    dataSummedByCustomer['percentageReturned'] = dataSummedByCustomer['returnQuantity'] / dataSummedByCustomer['quantity']
    dataSummedByCustomer = dataSummedByCustomer.drop(['returnQuantity', 'quantity'], 1)

    idToPercDict = dataSummedByCustomer.to_dict().get('percentageReturned')

    dataCopy.loc[:, 'percentageReturned'] = dataCopy['customerID'].apply(lambda custId: idToPercDict[custId])

    dataCopy = dataCopy.drop(['customerID'], 1)

    return dataCopy

def constructItemPercentageReturnColumn(data):
    # avoid chain indexing warning
    dataCopy = data.copy()

    dataByCustomer = dataCopy[['quantity', 'returnQuantity']].groupby(dataCopy['articleID'])

    dataSummedByCustomer = dataByCustomer.apply(sum)
    dataSummedByCustomer['itemPercentageReturned'] = dataSummedByCustomer['returnQuantity'] / dataSummedByCustomer['quantity']
    dataSummedByCustomer = dataSummedByCustomer.drop(['returnQuantity', 'quantity'], 1)

    idToPercDict = dataSummedByCustomer.to_dict().get('itemPercentageReturned')

    dataCopy.loc[:, 'itemPercentageReturned'] = data['articleID'].apply(lambda custId: idToPercDict[custId])

    dataCopy = dataCopy.drop(['articleID'], 1)

    return dataCopy

def addNewFeatures(data):

    #see whether the product was overpriced. price > recommended
    data.loc[:, 'overpriced'] = data['price'] > data['rrp']

    #see how much the data was discounted ( if price == 0, divide by 1 )
    data.loc[:, 'discountedAmount'] = data['voucherAmount'] / data['price'].apply(lambda pr: max(pr,1))

    data = constructPercentageReturnColumn(data)
    data = constructItemPercentageReturnColumn(data)
    return data


def performColorCodeEngineering(data):

    #get the thousands digit. in the color RAL system, that represents the "Base" color
    data['colorCode'] = data['colorCode'].apply(lambda code: code/1000)

    return data


def handleMissingValues(data):

    data = data.dropna()

    # ORRRR
    #
    # productGroupImputer = Imputer(missing_values='NaN', strategy='median')
    # data['productGroup'] = productGroupImputer.fit_transform(data['productGroup'])
    #
    # rrpImputer = Imputer(missing_values='NaN', strategy='mean')
    # data['rrp'] = rrpImputer.fit_transform(data['rrp'])
    #
    # #todo for the voucherID column, 6 values missing, decide the strategy for those. in mean time, drop them
    # data = data.dropna()
    #


    return data

def getFeatureEngineeredData(data, predictionColumnId = None, performOHE = True, performSizeCodeEng = True):

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

    keptColumns = ['colorCode', 'quantity', 'price', 'rrp','deviceID','paymentMethod','sizeCode','voucherAmount','customerID','articleID' ]

    if predictionColumnId:
        keptColumns.append(predictionColumnId)

    # avoid chain indexing warning
    data = data[keptColumns].copy()

    # construct additional features as a mixture of various ones
    data = addNewFeatures(data)

    # drop NAs
    data = handleMissingValues(data)


    #restrict prediction to 0/1 for now. Map everything greater than 1 to 1
    data['returnQuantity'] = data['returnQuantity'].apply(lambda retQuant: min(retQuant,1))

    if performOHE:
        data = performOHEOnColumn(data, 'deviceID')
        data = performOHEOnColumn(data, 'paymentMethod')
    
    if performSizeCodeEng:
        data = performSizeCodeEngineering(data)

    data = performColorCodeEngineering(data)

    # data = performDateEngineering(data, 'orderDate')

    print("\nKept columns {}".format(data.columns))

    return data


def getTrainAndTestData(data = None, performOHE = True, performSizeCodeEng = True):
    """
        returns train and test data based
        on input data frame. if None is passed,
        csv is automatically loaded.
    """
    if data is None:
        print("No data passed, reading CSV...")
        data = FileManager.getTrainingData()

    predictionColumnId = 'returnQuantity'

    data = getFeatureEngineeredData(data, predictionColumnId, performOHE, performSizeCodeEng)

    trainData, testData = train_test_split(data, test_size=0.25)

    xTrain = trainData.ix[:, trainData.columns != predictionColumnId].values
    yTrain = trainData[predictionColumnId].values

    xTest = testData.ix[:, testData.columns != predictionColumnId].values
    yTest = testData[predictionColumnId].values


    return xTrain,yTrain,xTest,yTest





















