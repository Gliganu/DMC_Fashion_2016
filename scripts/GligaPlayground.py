import time

import zaCode.ClassifierTrainer as ClassifierTrainer
import zaCode.DatasetManipulator as Toolbox
import zaCode.FileManager as FileManager
import zaCode.Validator as Validator
from zaCode import Visualizer


def addNewFeatures(data):

    data = Toolbox.constructOverpricedColumn(data)

    data = Toolbox.constructDiscountAmountColumn(data)

    data = Toolbox.constructArticleIdSuffixColumn(data)

    data = Toolbox.constructItemPercentageReturnColumn(data)

    data = Toolbox.constructBasketColumns(data)

    return data


def engineerOldFeatures(data):
    data = Toolbox.performOHEOnColumn(data, 'deviceID')

    data = Toolbox.performOHEOnColumn(data, 'paymentMethod')

    data = Toolbox.performSizeCodeEngineering(data)

    data = Toolbox.performColorCodeEngineering(data)

    data = Toolbox.performDateEngineering(data, 'orderDate')

    return data

def makePrediction():
    print("Reading data...")

    # data = FileManager.getRandomTrainingData(500000)
    # data = FileManager.get10kTrainingData()
    data = FileManager.getWholeTrainingData()

    keptColumns = ['orderDate', 'orderID', 'colorCode', 'quantity', 'price', 'rrp', 'deviceID', 'paymentMethod',
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
    polynomialFeaturesSourceColumns = ['quantity', 'price', 'voucherAmount', 'basketQuantity', 'itemPercentageReturned', 'overpriced',
                'discountedAmount']
    # polynomialFeaturesSourceColumns = data.columns
    data = Toolbox.constructPolynomialFeatures(data, polynomialFeaturesSourceColumns,degree=2, interaction_only=False)

    #Split into train/test data
    trainData, testData = Toolbox.performTrainTestSplit(data,0.25)

    #construct the percentage return column
    # trainData,testData = Toolbox.constructPercentageReturnColumn( trainData, testData )
    trainData,testData = Toolbox.constructCustomerMedianSizeAndColor(trainData, testData)

    trainData = trainData.drop(['customerID'], 1)
    testData = testData.drop(['customerID'], 1)

    print("\n\nFinal columns {} : {}".format(len(trainData.columns),trainData.columns))

    #X and Y train
    xTrain,yTrain = Toolbox.getXandYMatrix(trainData,'returnQuantity')
    # xTrain = Toolbox.scaleMatrix(xTrain)

    #Select K best features according to variance
    # xTrain, selectedColumns = Toolbox.selectKBest(xTrain, yTrain, 40, trainData.columns)
    # testData = testData[selectedColumns].copy()

    # X and Y test
    xTest, yTest = Toolbox.getXandYMatrix(testData, 'returnQuantity')
    # xTest = Toolbox.scaleMatrix(xTest)

    #Training the classifier
    classifier = ClassifierTrainer.trainClassifier(xTrain, yTrain)

    #Predicting
    yPred = classifier.predict(xTest)

    #Assessing the performance
    Validator.performValidation(yPred, yTest)
    Visualizer.plotFeatureImportance(classifier.feature_importances_,[col for col in trainData.columns if col != 'returnQuantity'])


if __name__ == '__main__':
    startTime = time.time()

    # play()
    makePrediction()

    # Visualizer.calculateLearningCurve(keptColumns)
    # Visualizer.calculateRocCurve()


    endTime = time.time()
    print("\nTotal run time:{}".format(endTime - startTime))