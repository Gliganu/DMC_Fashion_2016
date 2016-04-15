import pandas as pd



def getWholeTrainingData():

    inputFileName = '..\\data\\orders_train.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data

def getShortTrainingData():

    inputFileName = '..\\data\\orders_train_short.txt'
    data = pd.read_csv(inputFileName, delimiter=';', skipinitialspace=True)

    return data
