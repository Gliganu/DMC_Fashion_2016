import zaCode.ClassifierTrainer as ClassifierTrainer
import zaCode.DatasetManipulator as DatasetManipulator
import zaCode.Validator as Validator
import zaCode.FileManager as FileManager
import zaCode.Visualizer as Visualizer

import time
import numpy as np

def makePrediction():

    keptColumns = ['orderID', 'colorCode', 'quantity',
                                     'price', 'rrp', 'deviceID', 'paymentMethod',
                                     'sizeCode', 'voucherAmount', 'customerID', 'articleID']

    # construct Train & Test Data
    xTrain, yTrain, xTest, yTest = DatasetManipulator.getTrainAndTestData(keptColumns)

    # training the classifier
    classifier = ClassifierTrainer.trainClassifier(xTrain, yTrain)

    # predicting
    yPred = classifier.predict(xTest)

    # assessing the performance
    Validator.performValidation(yPred, yTest)


def play():
    data = FileManager.get10kTrainingData()

    # DatasetManipulator.constructPercentageReturnColumn(data)
#

if __name__ == '__main__':
    startTime = time.time()

    makePrediction()

    # play()
    # Visualizer.calculateLearningCurve()
    # Visualizer.calculateRocCurve()


    endTime = time.time()
    print("Total run time:{}".format(endTime - startTime))
