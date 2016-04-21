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
        
        # mask as True, needed only features in fest_condprob
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
    """
        returns data with sizeCode col normalised to best represent real clothing size
        note: renames column as normalisedSizeCode to prevent conflicts
        warning: modifies data frame passed in.
    """
    # No S, ai zis ca "should be reviewed", asa ca m-am uitat peste cod :D 
    # changes done:
    # 1.Normalised XS - XL from assigned values directly
    # 2.Deleted duplicate columns re-normalising A-I using chain indexing. 
    #   (maybe git merge fail? happened to me as well)
    # 3.Changed mean computation to exclude A and I values (which were initally == 2, and biasing mean)
    # 4.Eventually decided to write mean into A and I rows directly, skipping the = 2 part
    # 5.Renamed column to normalisedSizeCode to prevent clashes
    # Maybe we want to use .bool() and 'and' somehow instead of bitwise '&' ?
    # this is also an issue when using not(...), we need to use .apply(lambda v: not v) instead
    #   (even though bitwise and seems to work as well,
    #    we should test it works more rigurously and then it should be fine)
    
    # note initial code renormalised values but we can write
    # normalised values here directly
    data.loc[(data.sizeCode == 'XS'), 'sizeCode'] = 0
    data.loc[(data.sizeCode == 'S'), 'sizeCode'] = 1 / 4
    data.loc[(data.sizeCode == 'M'), 'sizeCode'] = 2 / 4
    data.loc[(data.sizeCode == 'L'), 'sizeCode'] = 3 / 4
    data.loc[(data.sizeCode == 'XL'), 'sizeCode'] = 4 / 4
    
    # get aux indexing and copy data
    notAorIindex = (data['sizeCode'] != 'A') & (data['sizeCode'] != 'I')
    numericData = pd.to_numeric(data[notAorIindex]['sizeCode']) # pd.to_numeric automatically copies
    
    # dropped 'sizeCode' indexing since numericData is now a single column
    # normalize 32-44
    numericData.loc[(numericData <= 44) & (numericData >= 32)] = \
        (numericData.loc[(numericData <= 44) & (numericData >= 32)] - 32.0) / (44.0 - 32.0)
    
    # normalize 24-33
    numericData.loc[(numericData <= 33) & (numericData >= 24)] = \
        (numericData.loc[(numericData <= 33) & (numericData >= 24)] - 24.0) / (33.0 - 24.0)
    
    # normalize 75-100
    numericData.loc[(numericData <= 100) & (numericData >= 75)] = \
        (numericData.loc[(numericData <= 100) & (numericData >= 75)] - 75.0) / (100.0 - 75.0)
    
    # writeback numericData into original dataframe
    # maybe we can work with data.loc in-place as the original did?
    # that means we should filter 'A' and 'I' out somehow first, 
    # otherwise <= and >= don't work for indexing
    data.loc[notAorIindex, 'sizeCode'] = numericData
    
    # set I and A to mean of the rest for the moment
    # apparently pandas complains about not(...) so I'm using the apply with lambda
    data.loc[notAorIindex.apply(lambda v: not v), 'sizeCode'] = data.loc[notAorIindex, 'sizeCode'].mean()
    
    # maybe update directly only 'sizeCode' location in list? O(n) anyways for find.
    data.columns = [ c if c != 'sizeCode' else 'normalisedSizeCode' for c in data.columns ]
    
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
    data = normalizeSize(data)
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
    dataSummedByCustomer['itemPercentageReturned'] = dataSummedByCustomer['returnQuantity'] / dataSummedByCustomer['quantity'].apply(lambda x: max(1,x))
    dataSummedByCustomer = dataSummedByCustomer.drop(['returnQuantity', 'quantity'], 1)

    idToPercDict = dataSummedByCustomer.to_dict().get('itemPercentageReturned')

    dataCopy.loc[:, 'itemPercentageReturned'] = data['articleID'].apply(lambda custId: idToPercDict[custId])

    dataCopy = dataCopy.drop(['articleID'], 1)

    return dataCopy


def constructPolynomialFeatures(data, sourceFeatures, degree=1, interaction_only=True, include_bias=False):
    print("Constructing polynomial features....")

    targetData = data[sourceFeatures].copy()

    # standardize everything
    dataMatrix = targetData.as_matrix().astype(np.float)
    scaler = StandardScaler()
    dataMatrix = scaler.fit_transform(dataMatrix)

    # construct polynomial features
    polynomialFeatures = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    newColumnsMatrix = polynomialFeatures.fit_transform(dataMatrix)

    newColumnsNames = []

    # construct the names of the newly generated sourceFeatures as we only have a matrix of numbers now
    for entry in polynomialFeatures.powers_:
        newFeature = []
        for feat, coef in zip(sourceFeatures, entry):
            if coef > 0:
                newFeature.append(feat + '^' + str(coef))
        if not newFeature:
            newColumnsNames.append("1")
        else:
            newColumnsNames.append(' + '.join(newFeature))

    newColumnsDataFrame = pd.DataFrame(newColumnsMatrix, columns=newColumnsNames)

    # drop all the sourceFeatures which are themselves to the power 1  ( as they already exist )
    newColumnsToBeDeleted = [featureName + "^1" for featureName in sourceFeatures]
    newColumnsDataFrame = newColumnsDataFrame.drop(newColumnsToBeDeleted, 1)

    finalData = data.join(newColumnsDataFrame)

    #todo one extra row added at the end. weird. to investigate
    finalData = finalData.dropna()

    return finalData


def constructOverpricedColumn(data):
    data.loc[:, 'overpriced'] = data['price'] > data['rrp']

    return data

def constructDiscountAmountColumn(data):
    data.loc[:, 'discountedAmount'] = data['voucherAmount'] / data['price'].apply(lambda pr: max(pr, 1))

    return data


def performColorCodeEngineering(data):
    # get the thousands digit. in the color RAL system, that represents the "Base" color
    data['colorCode'] = data['colorCode'].apply(lambda code: int(code) / 1000)

    return data


def dropMissingValues(data):
    return data.dropna()


def fillMissingValues(data, productGroupStategy = 'median', rrpStrategy = 'mean'):
    """
    Fills the missing values in the columns which suffer from this.
    """

    productGroupImputer = Imputer(missing_values='NaN', strategy=productGroupStategy)
    data['productGroup'] = productGroupImputer.fit_transform(data['productGroup'])

    rrpImputer = Imputer(missing_values='NaN', strategy=rrpStrategy)
    data['rrp'] = rrpImputer.fit_transform(data['rrp'])

    #for the voucher ID column, 6 values missing. drop them
    data = dropMissingValues(data)

    return data


def filterColumns(data, columnNames):
    data = data[columnNames].copy()

    print("\nKept columns ({}) : {} ".format(len(columnNames), columnNames))
    return data


def restrictReturnQuantityToBinaryChoice(data):
    data['returnQuantity'] = data['returnQuantity'].apply(lambda retQuant: min(retQuant, 1))

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
    # print("\nAfter Select K Best : Dismissed Features = {}".format(notSelectedColumnNames))

    return newXTrain,selectedColumnNames


def getXandYMatrix(data,predictionColumnName):
    """
    Being given a dataframe, it splits it's columns into X containing the training features and y being the prediciton feature
    """

    X = data.ix[:, data.columns != predictionColumnName].values
    y = data[predictionColumnName].values

    return X,y

def scaleMatrix(dataMatrix):
    """
    Scales all the columns in the matrix
    """

    scaler = StandardScaler()
    return scaler.fit_transform(dataMatrix)



def performTrainTestSplit(data, test_size):
    """
    Being given a dataframe and a testSize, it splits the data into training/test examples randomly according to the size provided
    """

    trainData, testData = train_test_split(data, test_size=test_size)

    return trainData, testData
