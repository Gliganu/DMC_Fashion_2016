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


def doRandom():
    data = FileManager.getWholeTrainingData();

    data  = data['colorCode'].unique()

    print data.shape

if __name__ == '__main__':

    startTime = time.time()

    performJob()

    # doRandom()
    # Visualizer.calculateLearningCurve()


    endTime = time.time()
    print("Total run time:{}".format(endTime - startTime))
