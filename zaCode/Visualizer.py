import zaCode.FileManager as FileManager
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import zaCode.DatasetManipulator as DatasetManipulator
import zaCode.ClassifierTrainer as ClassifierTrainer

from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.metrics import roc_curve, auc

#
#
# def plotReturnQuantityHist():
#
#     data = FileManager.getWholeTrainingData()
#
#     sns.distplot(data['returnQuantity'],kde=False)
#
#     plt.title("ReturnQuantity Histogram")
#     plt.show()
#
#
# def plotScatterPlots():
#
#     data = FileManager.getWholeTrainingData()
#
#     sns.jointplot(data['returnQuantity'],data['price'])
#
#     plt.title("ReturnQuantity x Price Scatterplot")
#     plt.show()
#
#
# def plotPairPlot():
#
#     data = FileManager.getWholeTrainingData()
#
#     # data = data[['colorCode','sizeCode','quantity','price','rrp','voucherAmount','returnQuantity']]
#     data = data[['colorCode','returnQuantity']]
#
#     sns.pairplot(data)
#
#     plt.title("Pairwise comparison of features")
#     plt.show()
#
#
#


def plotFeatureImportance(feature_importance,featureNames):

    # Normalize The Features
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(16, 12))
    plt.barh(pos, feature_importance[sorted_idx], align='center', color='#7A68A6')
    plt.yticks(pos, np.asanyarray(featureNames)[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

def calculateRocCurve():
    # construct Train & Test Data
    xTrain, yTrain, xTest, yTest = DatasetManipulator.getTrainAndTestData()

    # training the classifier
    classifier = ClassifierTrainer.trainClassifier(xTrain, yTrain)

    # predicting
    yPred = classifier.predict(xTest)


    false_positive_rate, true_positive_rate, thresholds = roc_curve(yTest, yPred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def calculateLearningCurve():

    data = FileManager.loadDataFrameFromCsv('allFeatures.csv',size=1000000)

    # Split into train/test data
    trainData, testData = DatasetManipulator.performTrainTestSplit(data, 0.25)

    # construct the percentage return column
    # trainData, testData = Toolbox.constructPercentageReturnColumn(trainData, testData, n_clusters=150)
    trainData = trainData.drop(['productGroup', 'deviceID', 'paymentMethod'], 1)
    testData = testData.drop(['productGroup', 'deviceID', 'paymentMethod'], 1)

    # consturct median color/size per customer + difference
    trainData, testData = DatasetManipulator.constructCustomerMedianSizeAndColor(trainData, testData)

    trainData = trainData.drop(['customerID'], 1)
    testData = testData.drop(['customerID'], 1)

    print("\n\nFinal columns {} : {}".format(len(trainData.columns), trainData.columns))

    # X and Y train
    xTrain, yTrain = DatasetManipulator.getXandYMatrix(trainData, 'returnQuantity')

    # X and Y test
    xTest, yTest = DatasetManipulator.getXandYMatrix(testData, 'returnQuantity')

    # for full dataset
    # trainSizes =  np.linspace(100000,1136593,5,dtype=int)

    # for 1 mil
    trainSizes =  np.linspace(100000,493055,5,dtype=int)


    classifier = ClassifierTrainer.trainClassifier(xTrain, yTrain)

    plot_learning_curve(classifier,xTrain,yTrain,trainSizes)



def plot_learning_curve(estimator, X, y,train_sizes):

    n_jobs = -1

    cv = cross_validation.ShuffleSplit(len(X), n_iter=1, test_size=0.3)

    plt.figure()

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,scoring="accuracy", cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,verbose=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    plt.show()
