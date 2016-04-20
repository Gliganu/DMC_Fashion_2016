import pandas as pd
import numpy as np
import math
import sys

from datetime import datetime
from collections import defaultdict
from copy import copy

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler, PolynomialFeatures, normalize
from sklearn.feature_selection import SelectKBest,f_regression

import zaCode.FileManager as FileManager


class DSetTransform:
    """ Class Transforms a  by dropping or replacing features
        Currently implements OHE, Conditional Probability and Dropping Cols.
        TODO Test.
    """
    
    def __init__(self, 
                 feats_kept = [],
                 feats_ohe = [],
                 feats_condprob = [],
                 target = 'returnQuantity'):
        
        self.feats_kept = feats_kept
        self.feats_ohe = feats_ohe
        self.feats_condprob = feats_condprob
        self.target = target
        
    def transformCondProb(self, data):
        """
            returns data with columns transformed according
            to feats_condprob set (if f in feats_condprob,
            column named f will be replaced with conditional
            probability P(target > 0 | f).
        """
        
        #prevent mutation of input data (dropOtherFeats also copies data frame)
        data_filtered = self.dropOtherFeats(data, self.feats_condprob)

        mask = defaultdict(lambda: False)
        mask.update({ f: True for f in self.feats_condprob })
        data_cprob, probMap = mapFeaturesToCondProbs(data_filtered, mask, self.target)
        
        return data_cprob
    
    def transformOHE(self, data):
        """
            returns data with binary columns for each categorical
            feature named in feats_ohe, and initial feature dropped
        """
        
        #prevent mutation of input data (dropOtherFeats also copies data frame)
        data = self.dropOtherFeats(data, self.feats_ohe)
        for f in self.feats_ohe:
            data = performOHEOnColumn(data, f)
        
        return data
    
    def dropOtherFeats(self, data, feats_kept = None):
        """
            returns new data frame containing only target varible
            and features contained in feats_kept
        """
        if feats_kept is None:
            feats_kept = self.feats_kept
            
        col_index = copy(feats_kept)
        col_index.append(self.target)
        
        return data[col_index].copy()
    
    def transform(self, data, drop = False):
        """
            transforms data frame with all operations, by default does not drop cols
        """
        # transform and drop target (we don't want duplicate cols)
        data_cprob = self.transformCondProb(data).drop([self.target], 1)
        data_ohe = self.transformOHE(data).drop([self.target], 1)

        data_filtered = self.dropOtherFeats(data) if drop else data.copy()

        return pd.concat([data_filtered, data_cprob, data_ohe], 1)
        
        
    
    
def mapFeaturesToCondProbs(rawData, featureMask=None, target='returnQuantity'):
    """
        replaces all features in the dataset with the
        conditional probability of the return qty being
        equal to or greater than one i.e. maps:
        feature -> P(returnQuantity > 0 | Feature = feature).

        param featureMask : dict of colName -> bool that
                            controls which feature is mapped
                            default = None, will map all
        param target : target variable to predict 
                       (in our case, returnQuantity)
        return modified data and global mapping used in tuple
    """
    if featureMask == None:
        featureMask = defaultdict(lambda: True, returnQuantity=False)

    # prevent mutation of input data
    rawData = rawData.copy()
    rawData.dropna()

    rowNum = len(rawData)
    # will also return a global map so we can do the inverse mappings
    globalMap = {}

    # cache data frame full of just favorable occurences
    valsFavorableDf = rawData[rawData[target] > 0].copy()

    for colName, series in rawData.iteritems():

        if featureMask[colName]:
            print('processing column ' + colName + '...')
            colMap = {}
            for uniqueVal in series.unique():
                allValsCnt = series[series == uniqueVal].count();
                valsFavorableSeries = (rawData[rawData[target] > 0])[colName]
                valsFavorableCnt = valsFavorableSeries[valsFavorableSeries == uniqueVal].count()

                colMap[uniqueVal] = valsFavorableCnt / allValsCnt

            globalMap[colName] = colMap

            # actually apply transform
            series = series.apply(lambda x: colMap[x])
            rawData[colName] = series
    
    #rename columns modified
    rawData.columns = [ (x if x == target or not featureMask[x] else "cprob_" + x) 
                        for x in rawData.columns ]
    
    return rawData, globalMap


def clusterRetQtyGTOne(rawData):
    """
        replaces all return qts greater than one
        with one, turning problem into binary classification.
    """
    rawData['returnQuantity'] = rawData['returnQuantity'].apply(lambda qty: 0 if qty < 1 else 1)

    return rawData


def normalizeSize(data):
    data.loc[(data.sizeCode == 'XS'), 'sizeCode'] = 0
    data.loc[(data.sizeCode == 'S'), 'sizeCode'] = 1
    data.loc[(data.sizeCode == 'M'), 'sizeCode'] = 2
    data.loc[(data.sizeCode == 'L'), 'sizeCode'] = 3
    data.loc[(data.sizeCode == 'XL'), 'sizeCode'] = 4
    # set A and I on 2 for now
    data.loc[(data.sizeCode == 'I'), 'sizeCode'] = 2
    data.loc[(data.sizeCode == 'A'), 'sizeCode'] = 2
    # set A and I on 2 for now
    data['sizeCode'][data['sizeCode'] == 'I'] = 2
    data['sizeCode'][data['sizeCode'] == 'A'] = 2
    data.loc[:, 'sizeCode'] = pd.to_numeric(data['sizeCode'])
    # normalize XS-XL
    data.loc[(data.sizeCode >= 0) & (data.sizeCode <= 4), 'sizeCode'] = \
        data.loc[(data.sizeCode >= 0) & (data.sizeCode <= 4), 'sizeCode'] / 4.0
    # normalize 32-44
    data.loc[(data.sizeCode <= 44) & (data.sizeCode >= 32), 'sizeCode'] = \
        (data.loc[(data.sizeCode <= 44) & (data.sizeCode >= 32), 'sizeCode'] - 32.0) / (44.0 - 32.0)
    # normalize 24-33
    data.loc[(data.sizeCode <= 33) & (data.sizeCode >= 24), 'sizeCode'] = \
        (data.loc[(data.sizeCode <= 33) & (data.sizeCode >= 24), 'sizeCode'] - 24.0) / (33.0 - 24.0)
    # normalize 75-100
    data.loc[(data.sizeCode <= 100) & (data.sizeCode >= 75), 'sizeCode'] = \
        (data.loc[(data.sizeCode <= 100) & (data.sizeCode >= 75), 'sizeCode'] - 75.0) / (100.0 - 75.0)
    # set I and A to mean for the moment
    data.loc[(data.sizeCode == 2), 'sizeCode'] = data['sizeCode'].mean()
    return data


def performSizeCodeEngineering(data):
    # drop everything that is not digit. About 200k examples ( maybe not the best way )
    # data = data[data['sizeCode'].apply(lambda x: x.isnumeric())]

    data = normalizeSize(data)
    return data


def constructBasketColumns(data):
    print("Constructing basket size and quantity features...")
    grouped_by_orderID = data['quantity'].groupby(data['orderID'])

    aggregated = pd.DataFrame()
    aggregated[['basketSize', 'basketTotalQuantity']] = grouped_by_orderID.agg([np.size, np.sum])

    dict_basket_total_quantity = aggregated.to_dict().get('basketTotalQuantity')
    dict_basket_size = aggregated.to_dict().get('basketSize')

    data['basketSize'] = data['orderID'].apply(lambda id: dict_basket_size[id])
    data['basketQuantity'] = data['orderID'].apply(lambda id: dict_basket_total_quantity[id])

    data = data.drop(['orderID'], 1)
    return data


def constructWeekDayColumn(data):
    dataCopy = data.copy()
    dataCopy['weekday'] = dataCopy['orderDate'].map(
        lambda date: 1 if datetime.strptime(date, '%Y-%m-%d').weekday() <= 4 else 0)
    return dataCopy


def performDateEngineering(rawData, dateColumn):
    # rawData[dateColumn+'-month']= rawData[dateColumn].map(lambda entryDate: float(entryDate.split("-")[1]))
    # rawData[dateColumn+'-day'] = rawData[dateColumn].map(lambda entryDate: float(entryDate.split("-")[2]))
    data = constructWeekDayColumn(rawData)
    data = data.drop([dateColumn], 1)

    return data


def performOHEOnColumn(data, columnName):
    """
        warning: may mutate your input data. cached a copy if needed!
    """
    
    # adding all the extra columns
    data = pd.concat([data, pd.get_dummies(data[columnName], prefix=columnName)], axis=1)

    # dropping the "source" column
    data = data.drop([columnName], 1)

    return data


def performSizeCodeEngineering(data):
    # drop everything that is not digit. About 200k examples ( maybe not the best way )
    data = data[data['sizeCode'].apply(lambda x: x.isdigit())]

    # avoid chain indexing warning
    return data.copy()


def constructPercentageReturnColumn(data):
    print("Constructing PercentageReturn feature....")

    # avoid chain indexing warning
    dataCopy = data.copy()

    dataByCustomer = dataCopy[['quantity', 'returnQuantity']].groupby(dataCopy['customerID'])

    dataSummedByCustomer = dataByCustomer.apply(sum)
    dataSummedByCustomer['percentageReturned'] = dataSummedByCustomer['returnQuantity'] / dataSummedByCustomer[
        'quantity']
    dataSummedByCustomer = dataSummedByCustomer.drop(['returnQuantity', 'quantity'], 1)

    idToPercDict = dataSummedByCustomer.to_dict().get('percentageReturned')

    dataCopy.loc[:, 'percentageReturned'] = dataCopy['customerID'].apply(lambda custId: idToPercDict[custId])

    dataCopy = dataCopy.drop(['customerID'], 1)

    return dataCopy


def constructItemPercentageReturnColumn(data):
    print("Constructing ItemPercentageReturn feature....")

    # avoid chain indexing warning
    dataCopy = data.copy()

    dataByCustomer = dataCopy[['quantity', 'returnQuantity']].groupby(dataCopy['articleID'])

    dataSummedByCustomer = dataByCustomer.apply(sum)
    dataSummedByCustomer['itemPercentageReturned'] = dataSummedByCustomer['returnQuantity'] / dataSummedByCustomer[
        'quantity']
    dataSummedByCustomer = dataSummedByCustomer.drop(['returnQuantity', 'quantity'], 1)

    idToPercDict = dataSummedByCustomer.to_dict().get('itemPercentageReturned')

    dataCopy.loc[:, 'itemPercentageReturned'] = data['articleID'].apply(lambda custId: idToPercDict[custId])

    dataCopy = dataCopy.drop(['articleID'], 1)

    return dataCopy


def constructPolynomialFeatures(data):
    print("Constructing polynomial features....")

    # get only the target columns
    features = ['quantity', 'price', 'voucherAmount', 'basketQuantity', 'percentageReturned', 'overpriced',
                'discountedAmount']

    targetData = data[features].copy()

    # standardize everything
    dataMatrix = targetData.as_matrix().astype(np.float)
    scaler = StandardScaler()
    dataMatrix = scaler.fit_transform(dataMatrix)

    # construct polynomial features
    polynomialFeatures = PolynomialFeatures(interaction_only=True, include_bias=False)
    newColumnsMatrix = polynomialFeatures.fit_transform(dataMatrix)

    newColumnsNames = []

    # construct the names of the newly generated features as we only have a matrix of numbers now
    for entry in polynomialFeatures.powers_:
        newFeature = []
        for feat, coef in zip(features, entry):
            if coef > 0:
                newFeature.append(feat + '^' + str(coef))
        if not newFeature:
            newColumnsNames.append("1")
        else:
            newColumnsNames.append(' + '.join(newFeature))

    newColumnsDataFrame = pd.DataFrame(newColumnsMatrix, columns=newColumnsNames)

    # drop all the features which are themselves to the power 1  ( as they already exist )
    newColumnsToBeDeleted = [featureName + "^1" for featureName in features]
    newColumnsDataFrame = newColumnsDataFrame.drop(newColumnsToBeDeleted, 1)

    data = data.join(newColumnsDataFrame)

    return data


def addNewFeatures(data):
    # see whether the product was overpriced. price > recommended
    data.loc[:, 'overpriced'] = data['price'] > data['rrp']

    # see how much the data was discounted ( if price == 0, divide by 1 )
    data.loc[:, 'discountedAmount'] = data['voucherAmount'] / data['price'].apply(lambda pr: max(pr, 1))

    data = constructPercentageReturnColumn(data)
    data = constructItemPercentageReturnColumn(data)
    data = constructBasketColumns(data)

    data = constructPolynomialFeatures(data)

    return data


def performColorCodeEngineering(data):
    # get the thousands digit. in the color RAL system, that represents the "Base" color
    data['colorCode'] = data['colorCode'].apply(lambda code: int(code) / 1000)

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


def getFeatureEngineeredData(data, keptColumns, predictionColumnId=None,  performOHE=True, performSizeCodeEng=True):
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

    if predictionColumnId:
        keptColumns.append(predictionColumnId)

    # avoid chain indexing warning
    data = data[keptColumns].copy()

    # construct additional features as a mixture of various ones
    data = addNewFeatures(data)

    # drop NAs
    data = handleMissingValues(data)

    # restrict prediction to 0/1 for now. Map everything greater than 1 to 1
    data['returnQuantity'] = data['returnQuantity'].apply(lambda retQuant: min(retQuant, 1))

    if performOHE:
        data = performOHEOnColumn(data, 'deviceID')
        data = performOHEOnColumn(data, 'paymentMethod')

    if performSizeCodeEng:
        data = performSizeCodeEngineering(data)

    data = performColorCodeEngineering(data)

    data = performDateEngineering(data, 'orderDate')

    print("\nKept columns ({}) : {} ".format(len(data.columns), data.columns))

    return data


def selectKBest(xTrain, yTrain, k, columnNames):
    """
    Select the K best features based on the variance they have along the training set
    """
    selector = SelectKBest(f_regression, k=k)  # k is number of features.
    newXTrain = selector.fit_transform(xTrain, yTrain)

    #print the remaining features
    columnNames = [name for name in columnNames if name != 'returnQuantity']

    selectedColumnNames = np.array(columnNames)[selector.get_support()]
    notSelectedColumnNames = np.array(columnNames)[np.invert(selector.get_support())]

    selectedColumnNames = np.append(selectedColumnNames, 'returnQuantity')

    print("\n\n\n\nAfter Select K Best : Nr features = {}".format(k))
    print("\nAfter Select K Best : Selected Features = {}".format(selectedColumnNames))
    print("\nAfter Select K Best : Dismissed Features = {}".format(notSelectedColumnNames))

    return newXTrain,selectedColumnNames


def performPostFeatureEngineering(trainData, testData, predictionColumnId, columnNames, selectK = False, standardize = False):
    """
     Performs various operations after all the features were created and filtered
                - Select K Best
                - Standardize all features
                - PCA ( to be implemented)
    """
    xTrain = trainData.ix[:, trainData.columns != predictionColumnId].values
    yTrain = trainData[predictionColumnId].values

    #select K best features
    if selectK:
        xTrain, selectedColumns = selectKBest(xTrain, yTrain, 20, columnNames)
        # after we select K best, we filter the testData as well to maintain only those columns
        testData = testData[selectedColumns].copy()

    xTest = testData.ix[:, testData.columns != predictionColumnId].values
    yTest = testData[predictionColumnId].values

    #standardizes all the values
    if standardize:
        scaler = StandardScaler()
        xTrain = scaler.fit_transform(xTrain)
        xTest = scaler.fit_transform(xTest)


    return xTrain,yTrain,xTest,yTest

#! @deprecated, should be replaced with a script for each different
#!              feature engineering run in stead of everybody modifying the 
#!              same function whenever they want a dataset.
#!              (put scripts in ../scriptRuns)
def getTrainAndTestData(keptColumns, data=None, performOHE=True, performSizeCodeEng=True):
    """
        returns train and test data based
        on input data frame. if None is passed,a
        csv is automatically loaded.
    """
    if data is None:
        print("No data passed, reading CSV...")
        data = FileManager.getWholeTrainingData()

    predictionColumnId = 'returnQuantity'

    data = getFeatureEngineeredData(data, keptColumns, predictionColumnId = predictionColumnId ,performOHE = performOHE, performSizeCodeEng = performSizeCodeEng)

    #split the data into training/test set by a specified ratio
    trainData, testData = train_test_split(data, test_size=0.25)

    xTrain,yTrain,xTest,yTest = performPostFeatureEngineering(trainData, testData, predictionColumnId, data.columns, selectK= False, standardize= False)

    return xTrain, yTrain, xTest, yTest
