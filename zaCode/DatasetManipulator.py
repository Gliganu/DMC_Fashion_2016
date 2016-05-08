import pandas as pd
import numpy as np
import math
import sys
import random
from datetime import datetime
from collections import defaultdict
from copy import copy
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler, PolynomialFeatures, normalize,Binarizer,LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.neural_network.rbm import BernoulliRBM
import zaCode.FileManager as FileManager
from sklearn.cluster import KMeans

class DSetTransform:
    """ Class Performs data set transformations for preprocessing
    """

    def __init__(self,
                 feats_kept=[],
                 feats_ohe=[],
                 feats_condprob=[],
                 feats_normed=[],
                 split_date=False,
                 target='returnQuantity'):

        self.feats_kept = feats_kept
        self.feats_ohe = feats_ohe
        self.feats_condprob = feats_condprob
        self.target = target
        self.feats_normed = feats_normed
        self.split_date_flag = split_date

    def norm_features(self, data, keep_target=True):
        """
        returns new data set with normalised features, and optional target column
        :param data:
        :param keep_target:
        :return:
        """

        retval = pd.DataFrame(columns=["normed_" + c for c in self.feats_normed])
        retval.loc[:, :] = data.loc[:, self.feats_normed]
        
        rows = len(data)
        for col in self.feats_normed:
            # compute mean
            m = 0
            cnt = 0
            print("computing " + col +" mean...")
            for idx in data.index:
                cnt += 1
                if cnt % 1e5 == 0:
                    print("passed elem " + str(cnt) + "...")
                m += data.loc[idx, col]
            m /= rows

            # compute stdev
            stdev = 0
            cnt = 0
            print("computing " + col + " stdev...")
            for idx in data.index:
                cnt += 1
                if cnt % 1e5 == 0:
                    print("passed elem " + str(cnt) + "...")
                
                tmp = m - data.loc[idx, col]
                stdev += tmp * tmp
            
            stdev /= rows
            stdev = math.sqrt(stdev)

            # normalise
            cnt = 0
            print("normalising " + col + "...")
            ncol = "normed_" + col
            for idx in data.index:
                cnt += 1
                if cnt % 1e5 == 0:
                    print("passed elem " + str(cnt) + "...")
                    
                retval.loc[idx, ncol] = (retval.loc[idx, ncol] - m) / stdev

        if keep_target:
            for idx in data.index:
                retval.loc[idx, self.target] = data.loc[idx, self.target]

        return retval

    def split_date(self, data, keep_target=True):
        """
        splits orderDate into orderYear,Month,Day
        :param data: dataset to be transformed (not mutated)
        :param keep_target: if true, also appends target variable in result data set
        :return: new data set with orderDate split cols and optional target col
        """

        new_cols = ["orderYear", "orderMonth", "orderDay" ]
        if keep_target:
            new_cols.append(self.target)
        retval = pd.DataFrame(columns = new_cols)

        if keep_target:
            for idx in data.index:
                ls = data.loc[idx, "orderDate"].split("-")
                retval.loc[idx, "orderYear"] = int(ls[0])
                retval.loc[idx, "orderMonth"] = int(ls[1])
                retval.loc[idx, "orderDay"] = int(ls[2])
                retval.loc[idx, self.target] = data.loc[idx, self.target]
        else:
            for idx in data.index:
                ls = data.loc[idx, "orderDate"].split("-")
                retval.loc[idx, "orderYear"] = int(ls[0])
                retval.loc[idx, "orderMonth"] = int(ls[1])
                retval.loc[idx, "orderDay"] = int(ls[2])

        return retval

    def periodic_partition(self, data, fraction):
        """
        Partitions dataset by selection periodic samples from timeseries
        :param data:
        :param fraction: fraction of data (in range [0, 1])
        :return: Single data frame containing selected subset
        """

        final_cnt = int(len(data) * fraction)
        print("selecting {} items from data".format(final_cnt))

        cnt = 0

        retval = pd.DataFrame(columns = data.columns)
        indicator = int(1.0 / fraction)
        for idx, vals in data.iterrows():
            if cnt % indicator == 0:
                retval.loc[idx, :] = vals
            cnt += 1

            if cnt % 20000 == 0:
                print("done with iteration number {}".format(cnt))

        return retval

    def partition(self, data, fraction):
        """ partitions dataset into two sets, containing fraction and 1-fraction 
            percentages of the data.
            0 <= fraction <= 1
        """
        # i'd use traint_test_split but that is designed for other purposes

        random.seed()  # systime used

        A = pd.DataFrame(columns=data.columns)
        B = pd.DataFrame(columns=data.columns)


        cnt = 0
        decision = False
        #take 10 products at a time to speedup subsampling

        for idx in data.index:
            if cnt % 5 == 0:
                decision = random.random() < fraction

            if decision:
                for cname, series in data.iteritems():
                    A.loc[idx, cname] = series[idx]
            else:
                for cname, series in data.iteritems():
                    B.loc[idx, cname] = series[idx]

            cnt += 1
            if cnt % 5000 == 0:
                print("processed {} rows".format(cnt))

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

        cols = [data_filtered, data_cprob, data_ohe]

        if self.split_date_flag:
            data_split = self.split_date(data, False) # false flag means do not keep target
            cols.append(data_split)

        if len(self.feats_normed) != 0:
            data_normed = self.norm_features(data, False) # false flag means do not keep target
            cols.append(data_normed)

        return pd.concat(cols, 1)


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


def performOHEOnColumn(data, columnName, withRemoval = True):
    """
        warning: may mutate your input data. cached a copy if needed!
    """

    # adding all the extra columns
    data = pd.concat([data, pd.get_dummies(data[columnName], prefix=columnName)], axis=1)

    # dropping the "source" column
    if withRemoval:
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

def getCustomerClusteringDataFrame(data):

    medianColumns = ['colorCode','productGroup','deviceID','paymentMethod']
    meanColumns = ['normalisedSizeCode','price','rrp','quantity']

    medianData = data[medianColumns].groupby(data['customerID'])
    meanData = data[meanColumns].groupby(data['customerID'])

    dataMedianByCustomer = medianData.median()
    dataMeanByCustomer = meanData.mean()

    clusteringTrainData = dataMedianByCustomer.join(dataMeanByCustomer)

    return clusteringTrainData

def getKnownCustomerIDToPercentageReturnDict(trainData):
    print("Constructing Known Customer ID To Percentage Returned ....")

    # avoid chain indexing warning
    trainDataCopy = trainData.copy()

    # construct the dictionary only on the information in the training set
    dataByCustomer = trainDataCopy[['quantity', 'returnQuantity']].groupby(trainDataCopy['customerID'])

    dataSummedByCustomer = dataByCustomer.apply(sum)
    dataSummedByCustomer['percentageReturned'] = dataSummedByCustomer['returnQuantity'] / dataSummedByCustomer[
        'quantity'].apply(lambda x: max(1, x))

    dataSummedByCustomer = dataSummedByCustomer.drop(['returnQuantity', 'quantity'], 1)

    customerIDtoPercentageReturnDict = dataSummedByCustomer.to_dict().get('percentageReturned')

    return customerIDtoPercentageReturnDict

def getFullCustomerIDToPercentageReturnDict(clusteringTrainData,clusteringTestData,knownCustomerIdToPercentageReturnDict,n_clusters):

    print("Clustering customers....")

    # compute the clusters based on the training data
    clusteringTrainDataValues = clusteringTrainData.values
    testDataCopy = clusteringTestData.copy()


    kMeans = KMeans(n_clusters=n_clusters)
    kMeans.fit(clusteringTrainDataValues)
    labels = kMeans.labels_

    #append the cluster index column to the dataframe
    trainDataCopy = clusteringTrainData.copy()
    trainDataCopy.loc[:, 'clusterIndex'] = labels


    trainDataCopy.loc[:, 'percentageReturned'] = trainDataCopy.index.map((lambda custId: knownCustomerIdToPercentageReturnDict[custId]))


    clusterLabelToPercentageReturnDict = {}

    #for each cluster, compute it's percentage return average based on the percReturn of the train data
    for i in range(n_clusters):
        customersInCluster = trainDataCopy[trainDataCopy['clusterIndex'] == i]
        average = customersInCluster['percentageReturned'].mean()
        clusterLabelToPercentageReturnDict[i] = average


    print("Predicting clusters for customers....")

    #todo for already seen customers do NOT predict !

    #predict in which cluster the entries in the test data will be
    predictedTestLabels = kMeans.predict(testDataCopy)
    testDataCopy.loc[:, 'clusterIndex'] = predictedTestLabels

    #set the percReturn of that entry to the mean of that belonging cluster
    testDataCopy.loc[:, 'percentageReturned'] = testDataCopy['clusterIndex'].apply(lambda clusterIndex: clusterLabelToPercentageReturnDict[clusterIndex])


    testCustomerIdToPercentageReturnDict = testDataCopy.to_dict().get('percentageReturned')

    #merge the 2 dictionaries
    knownCustomerIdToPercentageReturnDict.update(testCustomerIdToPercentageReturnDict)

    return knownCustomerIdToPercentageReturnDict


def constructPercentageReturnColumn(trainData,testData,n_clusters):

    print("Constructing Percentage Return Column....")

    trainDataCopy = trainData.copy()
    testDataCopy = testData.copy()

    #labelize the previously OHEed features
    paymendEncoder = LabelEncoder()
    deviceEncoder = LabelEncoder()

    paymendEncoder.fit(np.append(trainDataCopy['paymentMethod'],testDataCopy['paymentMethod']))
    deviceEncoder.fit(np.append(trainDataCopy['deviceID'],testDataCopy['deviceID']))

    trainDataCopy['paymentMethod'] = paymendEncoder.transform(trainDataCopy['paymentMethod'])
    testDataCopy['paymentMethod'] = paymendEncoder.transform(testDataCopy['paymentMethod'])

    trainDataCopy['deviceID'] = deviceEncoder.transform(trainDataCopy['deviceID'])
    testDataCopy['deviceID'] = deviceEncoder.transform(testDataCopy['deviceID'])


    clusteringTrainData = getCustomerClusteringDataFrame(trainDataCopy)
    clusteringTestData = getCustomerClusteringDataFrame(testDataCopy)

    knownCustomerIdToPercentageReturnDict = getKnownCustomerIDToPercentageReturnDict(trainDataCopy)

    fullCustomerIdToPercentageReturnDict = getFullCustomerIDToPercentageReturnDict(clusteringTrainData,
                                                                                    clusteringTestData,
                                                                                    knownCustomerIdToPercentageReturnDict,n_clusters)

    trainDataCopy.loc[:, 'percentageReturned'] = trainDataCopy['customerID'].apply(lambda custId: fullCustomerIdToPercentageReturnDict[custId])
    testDataCopy.loc[:, 'percentageReturned'] = testDataCopy['customerID'].apply(lambda custId: fullCustomerIdToPercentageReturnDict[custId])

    return trainDataCopy,testDataCopy

def constructBadPercentageReturnColumn(data):
    """DO NOT USE! It's the old bad percentage return which gives us optimistic results"""
    print("Constructing PercentageReturn feature....")

    # avoid chain indexing warning
    dataCopy = data.copy()

    dataByCustomer = dataCopy[['quantity', 'returnQuantity']].groupby(dataCopy['customerID'])

    dataSummedByCustomer = dataByCustomer.apply(sum)
    dataSummedByCustomer['percentageReturned'] = dataSummedByCustomer['returnQuantity'] / dataSummedByCustomer['quantity'].apply(lambda x: max(1,x))

    dataSummedByCustomer = dataSummedByCustomer.drop(['returnQuantity', 'quantity'], 1)

    idToPercDict = dataSummedByCustomer.to_dict().get('percentageReturned')

    dataCopy.loc[:, 'percentageReturned'] = dataCopy['customerID'].apply(lambda custId: idToPercDict[custId])

    return dataCopy



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

    #add new colums in the dataframes representing what color/size the customer usually buys
    trainDataCopy.loc[:, 'customerMedianColor'] = trainDataCopy['customerID'].apply(lambda custId: idToSize[custId])
    trainDataCopy.loc[:, 'customerMedianSize'] = trainDataCopy['customerID'].apply(lambda custId: idToColor[custId])

    testDataCopy.loc[:, 'customerMedianColor'] = testDataCopy['customerID'].apply(lambda custId: idToSize[custId])
    testDataCopy.loc[:, 'customerMedianSize'] = testDataCopy['customerID'].apply(lambda custId: idToColor[custId])


    #difference between what he bought now and what he normally buys
    trainDataCopy.loc[:, 'colorDifference'] = abs(trainDataCopy['customerMedianColor'] - trainDataCopy['colorCode'])
    trainDataCopy.loc[:, 'sizeDifference'] = abs(trainDataCopy['customerMedianSize'] - trainDataCopy['normalisedSizeCode'])

    testDataCopy.loc[:, 'colorDifference'] = abs(testDataCopy['customerMedianColor'] - testDataCopy['colorCode'])
    testDataCopy.loc[:, 'sizeDifference'] = abs(testDataCopy['customerMedianSize'] - testDataCopy['normalisedSizeCode'])

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


def contructOrderDuplicatesDistinctColorColumn(data):
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
    final = aggregated.drop([('colorCode', 'nunique')], axis=1)
    distinctColorDict = final.to_dict().get(('duplicateDistinctColor', ''))
    data['distinctColorDuplicate'] = data.apply(lambda row: distinctColorDict[(row['orderID'], row['articleID'])],
                                                axis=1)
    return dataCopy


def constructOrderDuplicatesDistinctSizeColumn(data):
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
    final = aggregated.drop([('sizeCode', 'nunique')], axis=1)
    distinctSizeDict = final.to_dict().get(('duplicateDistinctSize', ''))
    data['distinctSizeDuplicate'] = data.apply(lambda row: distinctSizeDict[(row['orderID'], row['articleID'])], axis=1)
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

def normalizeMatrix(dataMatrix):
    """
    Normalizes all the columns in the matrix
    """
    return normalize(dataMatrix)


def binarizeMatrix(dataMatrix, threshold):
    """
    Transforms all the inputs to either 0/1 . <0 Maps to 0. >1 Maps 1. [0,1] depends on the threshold you set between [0,1]
    """

    binarizer = Binarizer(threshold = threshold)

    dataMatrix = binarizer.fit_transform(dataMatrix)

    return dataMatrix

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


def performPCA(xTrain, xTest, numberComponents):

    print("Performing PCA...")

    pca = PCA(n_components = numberComponents)

    pca = pca.fit(xTrain)

    xTrain = pca.transform(xTrain)
    xTest = pca.transform(xTest)

    return xTrain, xTest


def performRBMTransform(xTrain, xTest):

    print("Performing RMB...")

    xTrain = normalize(xTrain)
    xTest = normalize(xTest)

    # xTrain = binarizeMatrix(xTrain,0.5)
    # xTest = binarizeMatrix(xTest,0.5)

    rmb = BernoulliRBM(verbose = True)

    rmb = rmb.fit(xTrain)

    xTrain = rmb.transform(xTrain)
    xTest = rmb.transform(xTest)

    return xTrain, xTest