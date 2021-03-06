import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import os

def getWholeTrainingData():
    inputFileName = '../data/orders_train.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data


def getTestData():
    inputFileName = '../data/orders_class.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data


def getRandomTrainingData(size):
    """
    Randomly selects "size" entries from the whole dataset.
    """
    data = getWholeTrainingData()
    train, _ = train_test_split(data, train_size=size)

    return train


def get10kTrainingData():
    inputFileName = '../data/orders_train_10k.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data


def get100kTrainingData():
    inputFileName = '../data/orders_train_100k.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data

def get500kTrainingData():
    inputFileName = '../data/orders_train_500k.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data

def get250kTrainingData():
    inputFileName = '../data/orders_train_250k.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data



def get1000kTrainingData():
    inputFileName = '../data/orders_train_1000k.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data

def getDistinctOHEFeatures():
    inputFileName = '../data/orders_distinct.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data

def saveModel(classifier, foldername, filename):
    if not os.path.exists('../models/' + foldername):
        os.makedirs('../models/' + foldername)
    joblib.dump(classifier, '../models/' + foldername + '/' + filename)

def loadModel(foldername, filename):
    return joblib.load('../models/' + foldername + '/' + filename)


def saveDataFrame(dataFrame, fileName, withIndex = False):
    if not os.path.exists('../dataframes'):
        os.makedirs('../dataframes')

    dataFrame.to_csv('../dataframes/' + fileName, sep='\t', index=withIndex)

def loadDataFrameFromCsv(fileName, size = None):

    inputFileName = '../dataframes/'+fileName

    data = pd.read_csv(inputFileName, delimiter='\t', skipinitialspace=True)

    #get random somple of size = size
    if size:
        data, _ = train_test_split(data, train_size=size)

    return data

