import pandas as pd
import numpy as np
import math
import sys
import random
from datetime import datetime
from collections import defaultdict
from copy import copy
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler, PolynomialFeatures, normalize
from sklearn.feature_selection import SelectKBest, f_regression
import zaCode.FileManager as FileManager


class DSetTransform:
    """ Class Transforms a  by dropping or replacing features and partitioning data set
        Currently implements OHE, Conditional Probability and Dropping Cols.
    """

    def __init__(self,
                 feats_kept=[],
                 feats_ohe=[],
                 feats_condprob=[],
                 target='returnQuantity'):

        self.feats_kept = feats_kept
        self.feats_ohe = feats_ohe
        self.feats_condprob = feats_condprob
        self.target = target

    def partition(self, data, fraction):
        """ partitions dataset into two sets, containing fraction and 1-fraction 
            percentages of the data.
            0 <= fraction <= 1
        """
        # i'd use traint_test_split but that is designed for other purposes

        random.seed()  # systime used

        A = pd.DataFrame(columns=data.columns)
        B = pd.DataFrame(columns=data.columns)

        for idx in data.index:

            if random.random() < fraction:
                for cname, series in data.iteritems():
                    A.loc[idx, cname] = series[idx]
            else:
                for cname, series in data.iteritems():
                    B.loc[idx, cname] = series[idx]

        return A, B

    def transformCondProb(self, data):
        """
            returns data with columns transformed according
            to feats_condprob set (if f in feats_condprob,
            column named f will be replaced with conditional
            probability P(target > 0 | f).
        """

        # prevent mutation of input data (dropOtherFeats also copies data frame)
        data_filtered = self.dropOtherFeats(data, self.feats_condprob)

        # mask as True, needed only features in fest_condprob
        mask = defaultdict(lambda: False)
        mask.update({f: True for f in self.feats_condprob})

        data_cprob, probMap = mapFeaturesToCondProbs(data_filtered, mask, self.target)

        return data_cprob

    def transformOHE(self, data):
        """
            returns data with binary columns for each categorical
            feature named in feats_ohe, and initial feature dropped
        """

        # prevent mutation of input data (dropOtherFeats also copies data frame)
        data = self.dropOtherFeats(data, self.feats_ohe)
        for f in self.feats_ohe:
            data = performOHEOnColumn(data, f)

        return data

    def dropOtherFeats(self, data, feats_kept=None):
        """
            returns new data frame containing only target varible
            and features contained in feats_kept
        """
        if feats_kept is None:
            feats_kept = self.feats_kept

        col_index = copy(feats_kept)
        col_index.append(self.target)

        return data[col_index].copy()

    def transform(self, data, drop=False):
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

    # rename columns modified
    rawData.columns = [(x if x == target or not featureMask[x] else "cprob_" + x)
                       for x in rawData.columns]

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
    numericData = pd.to_numeric(data[notAorIindex]['sizeCode'])  # pd.to_numeric automatically copies

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
    data.columns = [c if c != 'sizeCode' else 'normalisedSizeCode' for c in data.columns]

    return data


def constructBasketColumns(data):
    print("Constructing BasketSize and BasketQuantity features...")
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
    data = constructWeekDayColumn(rawData)
    data[dateColumn + '-month'] = rawData[dateColumn].map(lambda entryDate: float(entryDate.split("-")[1]))
    data[dateColumn + '-day'] = rawData[dateColumn].map(lambda entryDate: float(entryDate.split("-")[2]))
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


def printNumberOfCustomersSeen(trainData, testData):
    trainCustomer = pd.Series(trainData['customerID'].unique())
    testCustomer = pd.Series(testData['customerID'].unique())

    totalTrainCustomer = trainCustomer.size
    seenCustomersNumber = trainCustomer.isin(testCustomer).sum()

    print("Percentage of already seen customers in test set: {}".format(seenCustomersNumber / totalTrainCustomer))


def constructPercentageReturnColumn(trainData, testData):
    print("Constructing PercentageReturn feature....")

    printNumberOfCustomersSeen(trainData, testData)

    # avoid chain indexing warning
    trainDataCopy = trainData.copy()
    testDataCopy = testData.copy()

    # construct the dictionary only on the information in the training set
    dataByCustomer = trainDataCopy[['quantity', 'returnQuantity']].groupby(trainDataCopy['customerID'])

    dataSummedByCustomer = dataByCustomer.apply(sum)
    dataSummedByCustomer['percentageReturned'] = dataSummedByCustomer['returnQuantity'] / dataSummedByCustomer[
        'quantity'].apply(lambda x: max(1, x))

    dataSummedByCustomer = dataSummedByCustomer.drop(['returnQuantity', 'quantity'], 1)

    # computing the average percentage across all customers
    summedByCustomerDict = dataSummedByCustomer.to_dict().get('percentageReturned')
    percentagesList = list(summedByCustomerDict.values())
    averagePercetangeOfReturn = sum(percentagesList) / float(len(percentagesList))

    # if customer not found, the default percentage will be the average of percetages
    idToPercDict = defaultdict(lambda: averagePercetangeOfReturn)

    # append the other dictionary
    idToPercDict.update(summedByCustomerDict)

    trainDataCopy.loc[:, 'percentageReturned'] = trainDataCopy['customerID'].apply(lambda custId: idToPercDict[custId])
    testDataCopy.loc[:, 'percentageReturned'] = testDataCopy['customerID'].apply(lambda custId: idToPercDict[custId])

    return trainDataCopy, testDataCopy


def constructCustomerMedianSizeAndColor(trainData, testData):
    trainDataCopy = trainData.copy()
    testDataCopy = testData.copy()

    trainDataCopy['normalisedSizeCode'] = pd.to_numeric(trainDataCopy['normalisedSizeCode'])

    trainDataCopy = np.round(trainDataCopy, 2)

    groupedByCustomer = trainDataCopy[['normalisedSizeCode', 'colorCode']].groupby(trainDataCopy['customerID'])

    median = groupedByCustomer.median()

    # if customer not found, the default will be 0
    idToSize = defaultdict(lambda: 0)
    idToColor = defaultdict(lambda: 0)

    # append the other dictionary
    idToSize.update(median.to_dict().get('normalisedSizeCode'))
    idToColor.update(median.to_dict().get('colorCode'))

    # add new colums in the dataframes
    trainDataCopy.loc[:, 'customerMedianColor'] = trainDataCopy['customerID'].apply(lambda custId: idToSize[custId])
    trainDataCopy.loc[:, 'customerMedianSize'] = trainDataCopy['customerID'].apply(lambda custId: idToColor[custId])

    testDataCopy.loc[:, 'customerMedianColor'] = testDataCopy['customerID'].apply(lambda custId: idToSize[custId])
    testDataCopy.loc[:, 'customerMedianSize'] = testDataCopy['customerID'].apply(lambda custId: idToColor[custId])

    return trainDataCopy, testDataCopy


def constructItemPercentageReturnColumn(data):
    print("Constructing ItemPercentageReturn feature....")

    # avoid chain indexing warning
    dataCopy = data.copy()

    dataByCustomer = dataCopy[['quantity', 'returnQuantity']].groupby(dataCopy['articleID'])

    dataSummedByCustomer = dataByCustomer.apply(sum)
    dataSummedByCustomer['itemPercentageReturned'] = dataSummedByCustomer['returnQuantity'] / dataSummedByCustomer[
        'quantity'].apply(lambda x: max(1, x))
    dataSummedByCustomer = dataSummedByCustomer.drop(['returnQuantity', 'quantity'], 1)

    idToPercDict = dataSummedByCustomer.to_dict().get('itemPercentageReturned')

    dataCopy.loc[:, 'itemPercentageReturned'] = data['articleID'].apply(lambda custId: idToPercDict[custId])

    dataCopy = dataCopy.drop(['articleID'], 1)

    return dataCopy


def constructPolynomialFeatures(data, sourceFeatures, degree=1, interaction_only=True, include_bias=False):
    """"
    THIS HAS A BUG ! DON'T USE FOR NOW.
    """
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

    # todo one extra row added at the end. weird. to investigate
    finalData = finalData.dropna()

    return finalData


def constructOrderDuplicatesCountColumn(data):
    print("Constructing order duplicates count feature")
    dataCopy = data.copy()
    # select only columns of interest - orderID and articleID
    filtered = dataCopy.loc[:, ['orderID', 'articleID']]
    # group by orderID
    groupedByOrder = filtered.groupby('orderID')
    # apply two aggregate functions - size -> number of instances in one order
    # nunique ->  number of unique instances (i.e. unique articleIDs) in one order
    aggregated = groupedByOrder.agg([np.size, pd.Series.nunique])
    # difference between size and nunique will give us number of duplicate articleIDs in one order
    aggregated['duplicates'] = aggregated.loc[:, ('articleID', 'size')] - aggregated.loc[:, ('articleID', 'nunique')]
    finalFrame = aggregated.drop([('articleID', 'size'), ('articleID', 'nunique')], 1)
    # turn the df into a dict with key = orderID, value = number of duplicates
    dictDuplicatesCount = finalFrame.to_dict().get(('duplicates', ''))
    # add a duplicates/order column in the original df, using the generated dict
    dataCopy.loc[:, 'orderDuplicatesCount'] = data['orderID'].apply(lambda orderID: dictDuplicatesCount[orderID])
    return dataCopy


def contructOrderDuplicatesDistinctColorCountColumn(data):
    print("Constructing order duplicate with distinct color count feature")
    dataCopy = data.copy()
    # select only the columns we need for the feature construction
    filteredOrderArticle = dataCopy.loc[:, ['orderID', 'articleID', 'colorCode']]
    # nested group by orderID and then by articleID
    groupedByOrderArticle = filteredOrderArticle.groupby(['orderID', 'articleID'])
    # get unique color code for each articleID in each order
    aggregated = groupedByOrderArticle.agg([pd.Series.nunique])
    # if nunique is 1, duplicateDistinctColor must be 0 (we either have non-duplicate article, hence 1 unique color,
    # or duplicate article, with one unique color)
    aggregated['duplicateDistinctColor'] = aggregated[('colorCode', 'nunique')].apply(lambda x: 0 if x == 1 else 1)
    # reset indices - want to have orderID as column, not index, in order to perform yet another group by
    aggregated.reset_index(inplace=True)
    # drop unused columns - will have just orderID and count of unique color codes per article
    final = aggregated.drop([('articleID', ''), ('colorCode', 'nunique')], axis=1)
    # group again by order id
    groupedByOrderFinal = final.groupby('orderID')
    # sum up
    finalDuplicateColors = groupedByOrderFinal.agg([np.sum], level=0)
    duplicateWithDistinctColorDict = finalDuplicateColors.to_dict().get(('duplicateDistinctColor', '', 'sum'))
    dataCopy['orderDuplicatesDistinctColorCount'] = dataCopy['orderID'].apply(
        lambda orderId: duplicateWithDistinctColorDict.get(orderId))
    return dataCopy


def constructOrderDuplicatesDistinctSizeCountColumn(data):
    print("Constructing order duplicate with distinct size count feature")
    dataCopy = data.copy()
    # select only the columns we need for the feature construction
    filteredOrderArticle = dataCopy[['orderID', 'articleID', 'sizeCode']]
    # nested group by orderID and then by articleID
    groupedByOrderArticle = filteredOrderArticle.groupby(['orderID', 'articleID'])
    # get unique size code for each articleID in each order
    aggregated = groupedByOrderArticle.agg([pd.Series.nunique])
    # if nunique is 1, duplicateDistinctColor must be 0 (we either have non-duplicate article, hence 1 unique size,
    # or duplicate article, with one unique size)
    aggregated['duplicateDistinctSize'] = aggregated[('sizeCode', 'nunique')].apply(lambda x: 0 if x == 1 else 1)
    # reset indices - want to have orderID as column, not index, in order to perform yet another group by
    aggregated.reset_index(inplace=True)
    # drop unused columns - will have just orderID and count of unique size codes per article
    final = aggregated.drop([('articleID', ''), ('sizeCode', 'nunique')], axis=1)
    # group again by order id
    groupedByOrderFinal = final.groupby('orderID')
    # sum up
    finalDuplicateSizes = groupedByOrderFinal.agg([np.sum], level=0)
    duplicateWithDistinctSizeDict = finalDuplicateSizes.to_dict().get(('duplicateDistinctSize', '', 'sum'))
    dataCopy['orderDuplicatesDistinctSizeCount'] = dataCopy['orderID'].apply(
        lambda orderId: duplicateWithDistinctSizeDict.get(orderId))
    return dataCopy


def constructOverpricedColumn(data):
    data.loc[:, 'overpriced'] = data['price'] > data['rrp']

    return data


def constructDiscountAmountColumn(data):
    data.loc[:, 'discountedAmount'] = data['voucherAmount'] / data['price'].apply(lambda pr: max(pr, 1))

    return data


def performColorCodeEngineering(data):
    # get the thousands digit. in the color RAL system, that represents the "Base" color
    data['colorCode'] = data['colorCode'].apply(lambda code: int(code / 1000))

    return data


def constructArticleIdSuffixColumn(data):
    data['articleIdSuffix'] = data['articleID'].apply(lambda id: int(id[4:]))

    return data


def dropMissingValues(data):
    return data.dropna()


def fillMissingValues(data, productGroupStategy='median', rrpStrategy='mean'):
    """
    Fills the missing values in the columns which suffer from this.
    """

    productGroupImputer = Imputer(missing_values='NaN', strategy=productGroupStategy)
    data['productGroup'] = productGroupImputer.fit_transform(data['productGroup'])

    rrpImputer = Imputer(missing_values='NaN', strategy=rrpStrategy)
    data['rrp'] = rrpImputer.fit_transform(data['rrp'])

    # for the voucher ID column, 6 values missing. drop them
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

    # print the remaining features
    columnNames = [name for name in columnNames if name != 'returnQuantity']

    selectedColumnNames = np.array(columnNames)[selector.get_support()]
    notSelectedColumnNames = np.array(columnNames)[np.invert(selector.get_support())]

    selectedColumnNames = np.append(selectedColumnNames, 'returnQuantity')

    print("\n\n\n\nAfter Select K Best : Nr features = {}".format(k))
    print("\nAfter Select K Best : Selected Features = {}".format(selectedColumnNames))
    # print("\nAfter Select K Best : Dismissed Features = {}".format(notSelectedColumnNames))

    return newXTrain, selectedColumnNames


def getXandYMatrix(data, predictionColumnName):
    """
    Being given a dataframe, it splits it's columns into X containing the training features and y being the prediciton feature
    """

    X = data.ix[:, data.columns != predictionColumnName].values
    y = data[predictionColumnName].values

    return X, y


def scaleMatrix(dataMatrix):
    """
    Scales all the columns in the matrix
    """

    scaler = StandardScaler()
    return scaler.fit_transform(dataMatrix)


def performTrainTestSplit(data, test_size):
    """
    Being given a dataframe and a testSize, it splits the data into training/test examples according to the size provided

    Test size is percent !!

    Returns them in order !!
    """

    numberTraining = math.ceil(data.shape[0] * (1 - test_size))
    trainData = data.iloc[0: int(numberTraining)]
    testData = data.iloc[int(numberTraining):]

    return trainData, testData
