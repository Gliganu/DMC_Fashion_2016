import FileManager
import Visualizer
import ClassifierTrainer
import DatasetManipulator
import Validator

import time

def performJob():
    # construct Train & Test Data
    xTrain, yTrain, xTest, yTest = DatasetManipulator.getTrainAndTestData()

    # training the classifier
    classifier = ClassifierTrainer.trainClassifier(xTrain, yTrain)

    # predicting
    yPred = classifier.predict(xTest)

    # assessing the performance
    Validator.performValidation(yPred, yTest)


if __name__ == '__main__':

    # startTime = time.time()
    # performJob()
    # endTime = time.time()
    # print("Total run time:{}".format(endTime - startTime))
    data = FileManager.getWholeTrainingData()
    DatasetManipulator.normalizeSize(data)
