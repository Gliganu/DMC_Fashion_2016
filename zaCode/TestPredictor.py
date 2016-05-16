import pandas as pd
import zaCode.Postprocesser as postprocesser
import zaCode.FileManager as filemanager


def makePrediction(classifierUnseenCl, classifierSeenCl, test, customerDict, dropUnseen, dropSeen):
    testWithPred = test.copy()
    for index, row in test.iterrows():
        if row['customerID'] in customerDict:
            prediction = classifierSeenCl.predict(test.loc[index, :].drop(dropSeen).reshape(1, -1))
        else:
            prediction = classifierUnseenCl.predict(
                test.loc[index, :].drop(dropUnseen).reshape(1, -1))

        testWithPred.loc[index, 'prediction'] = prediction
    # can use it when will have all the data in the test set (eg.article ID)
    # test = postprocesser.postprocess(test)
    filemanager.saveDataFrame(testWithPred, "test_result.csv")
    return testWithPred.loc[:, 'prediction']
