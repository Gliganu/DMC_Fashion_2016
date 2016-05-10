import time

import zaCode.ClassifierTrainer as ClassifierTrainer
import zaCode.DatasetManipulator as Toolbox
import zaCode.FileManager as FileManager
import zaCode.Validator as Validator
from zaCode import Visualizer
from sklearn.ensemble import VotingClassifier


def addNewFeatures(data):
    data = Toolbox.constructOverpricedColumn(data)

    data = Toolbox.constructDiscountAmountColumn(data)

    data = Toolbox.constructOrderDuplicatesCountColumn(data)

    data = Toolbox.contructOrderDuplicatesDistinctColorColumn(data)

    data = Toolbox.constructOrderDuplicatesDistinctSizeColumn(data)

    data = Toolbox.constructArticleIdSuffixColumn(data)
    # data = data.drop(['articleID'],1)

    data = Toolbox.constructItemPercentageReturnColumn(data)

    data = Toolbox.constructBasketColumns(data)
    # data = data.drop(['orderID'], 1)

    return data


def engineerOldFeatures(data):
    data = Toolbox.performOHEOnColumn(data, 'deviceID', False)
    # data = data.drop(['deviceID'],1)

    data = Toolbox.performOHEOnColumn(data, 'paymentMethod', False)
    # data = data.drop(['paymentMethod'], 1)

    data = Toolbox.performSizeCodeEngineering(data)
    # data = data.drop(['sizeCode'], 1)

    # data = Toolbox.performColorCodeEngineering(data)

    data = Toolbox.performDateEngineering(data, 'orderDate')
    # data = data.drop(['orderDate'], 1)

    return data


def serializeDataFrame():

    fileName =  'allFeatures.csv'

    data = constructDataFromScratch()

    FileManager.saveDataFrame(data,fileName)

def constructDataFromScratch():

    # data = FileManager.getRandomTrainingData(1000)
    data = FileManager.getRandomTrainingData(50000)
    # data = FileManager.get10kTrainingData()
    data = FileManager.getWholeTrainingData()

    keptColumns = ['orderDate', 'orderID', 'colorCode', 'quantity', 'price', 'rrp', 'deviceID', 'paymentMethod',
                   'productGroup',
                   'sizeCode', 'voucherAmount', 'customerID', 'articleID', 'returnQuantity']

    # Keep just the columns of interest
    data = Toolbox.filterColumns(data, keptColumns)

    # Restrict prediction to 0/1 for now. Map everything greater than 1 to 1
    data = Toolbox.restrictReturnQuantityToBinaryChoice(data)

    # Handle missing values
    data = Toolbox.dropMissingValues(data)

    # Construct additional features
    data = addNewFeatures(data)

    # Perform feature engineering on existing features
    data = engineerOldFeatures(data)

    # Construct polynomial features
    # polynomialFeaturesSourceColumns = ['quantity', 'price', 'voucherAmount', 'basketQuantity', 'itemPercentageReturned', 'overpriced', 'discountedAmount']
    # data = Toolbox.constructPolynomialFeatures(data, polynomialFeaturesSourceColumns,degree=2, interaction_only=False)

    return data




def makePrediction():
    print("Reading data...")

    data = constructDataFromScratch()

    # data = constructDataFromScratch()
    data = FileManager.loadDataFrameFromCsv('allFeatures.csv')


    # Split into train/test data
    trainData, testData = Toolbox.performTrainTestSplit(data, 0.25)

    # construct the percentage return column
    trainData, testData = Toolbox.constructPercentageReturnColumn(trainData, testData, n_clusters=150)
    trainData = trainData.drop(['productGroup', 'deviceID', 'paymentMethod'], 1)
    testData = testData.drop(['productGroup', 'deviceID', 'paymentMethod'], 1)

    # consturct median color/size per customer + difference
    trainData, testData = Toolbox.constructCustomerMedianSizeAndColor(trainData, testData)

    trainData = trainData.drop(['customerID'], 1)
    testData = testData.drop(['customerID'], 1)

    print("\n\nFinal columns {} : {}".format(len(trainData.columns), trainData.columns))

    # X and Y train
    xTrain, yTrain = Toolbox.getXandYMatrix(trainData, 'returnQuantity')

    # Select K best features according to variance
    # xTrain, selectedColumns = Toolbox.selectKBest(xTrain, yTrain, 40, trainData.columns)
    # testData = testData[selectedColumns].copy()

    # X and Y test
    xTest, yTest = Toolbox.getXandYMatrix(testData, 'returnQuantity')

    # xTrain = Toolbox.normalize(xTrain)
    # xTest = Toolbox.normalize(xTest)

    # apply PCA
    # xTrain,xTest = Toolbox.performPCA(xTrain,xTest,10)

    # apply RMB Transform ( + normalize and binarize )
    # xTrain,xTest = Toolbox.performRBMTransform(xTrain,xTest)

    # Training the classifier
    # classifier = ClassifierTrainer.trainClassifier(xTrain, yTrain)

    # FileManager.saveModel(classifier, 'gliga/randomForest', 'GligaRandomForest.pkl')
    logisticRegressionClassifier = FileManager.loadModel('gliga/logisticRegression', 'GligaLogisticRegression.pkl')
    gradientBoostingClassifier = FileManager.loadModel('gliga/gradientBoosting', 'GligaGradientBoosting.pkl')
    randomForestClassifier = FileManager.loadModel('gliga/randomForest', 'GligaRandomForest.pkl')
    naiveBayesClassifier = FileManager.loadModel('gliga/naiveBayes', 'GligaNaiveBayes.pkl')

    # classifier = VotingClassifier(estimators=[('lr', logisticRegressionClassifier), ('gb', gradientBoostingClassifier),
    #                                           ('rf', randomForestClassifier), ('nb', naiveBayesClassifier)],
    #                               voting='hard')
    # FileManager.saveModel(classifier, 'gliga_full/lr1', 'logistic.pkl')

    # randomForestClassifier = FileManager.loadModel('gliga/randomForest', 'GligaRandomForest.pkl')

    # Predicting
    predictionMatrix = Toolbox.constructPredictionMatrix(xTest, logisticRegressionClassifier,
                                                         gradientBoostingClassifier, randomForestClassifier,
                                                         naiveBayesClassifier)
    yPred = Toolbox.makeHardVoting(predictionMatrix)

    # Assessing the performance
    Validator.performValidation(yPred, yTest)
    # Visualizer.plotFeatureImportance(classifier.feature_importances_,[col for col in trainData.columns if col != 'returnQuantity'])


if __name__ == '__main__':
    startTime = time.time()

    # makePrediction()
    # serializeDataFrame()

    Visualizer.calculateLearningCurve()
    # Visualizer.calculateRocCurve()


    endTime = time.time()
    print("\nTotal run time:{}".format(endTime - startTime))
