import pandas as pd


def getWholeTrainingData():
    inputFileName = '../data/orders_train.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data


def getTestData():
    inputFileName = '../data/orders_class.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data


# added these 'peasanty' functions becauuse although we could've read the whole csv and trim the data frame after, it's faster this way
def get1kTrainingData():
    inputFileName = '../data/orders_train_1k.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data


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
