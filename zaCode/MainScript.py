import zaCode.ClassifierTrainer as ClassifierTrainer
import zaCode.DatasetManipulator as DatasetManipulator
import zaCode.Validator as Validator

import time


def makePrediction():
    # construct Train & Test Data
    xTrain, yTrain, xTest, yTest = DatasetManipulator.getTrainAndTestData()

    # training the classifier
    classifier = ClassifierTrainer.trainClassifier(xTrain, yTrain)

    # predicting
    yPred = classifier.predict(xTest)

    # assessing the performance
    Validator.performValidation(yPred, yTest)

if __name__ == '__main__':
    startTime = time.time()

    makePrediction()

    # Visualizer.calculateLearningCurve()


    endTime = time.time()
    print("Total run time:{}".format(endTime - startTime))
