import pandas as pd
import zaCode.Postprocesser as postprocesser
import FileManager as filemanager

def makePrediction(classifierSeenCl, classifierUnseenCl, test, customerDict):
    testForSeen = test.drop(['orderID', 'articleID'], 1)
    testForUnseen = test.drop(['orderID', 'articleID', 'percentageReturned'], 1)
    for index, row in test.iterrows():
        if row['customerID'] in customerDict:
            prediction = classifierSeenCl.predict(testForSeen[:][index])
        else:
            prediction = classifierUnseenCl.predict(testForUnseen[:][index])
        test.loc[index, 'returnQuantity'] = prediction
    test = postprocesser.postprocess(test)
    filemanager.saveDataFrame(test, "test_result.csv")
