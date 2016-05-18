import pandas as pd
import zaCode.Postprocesser as postprocesser
import zaCode.FileManager as filemanager


def makePrediction(classifierUnseenCl, classifierSeenCl, test, customerDict, dropUnseen, dropSeen):
    testWithPred = test.copy()

    seen = 0
    unseen = 0
    i = 0
    for index, row in test.iterrows():

        i+=1
        if i % 100 == 0:
            print("Predicting for {}".format(i))

        if row['customerID'] in customerDict:
            prediction = classifierSeenCl.predict(test.loc[index, :].drop(dropSeen).reshape(1, -1))
            seen +=1
        else:
            prediction = classifierUnseenCl.predict(
                test.loc[index, :].drop(dropUnseen).reshape(1, -1))
            unseen+=1

        testWithPred.loc[index, 'prediction'] = prediction
    # can use it when will have all the data in the test set (eg.article ID)
    # test = postprocesser.postprocess(test)
    filemanager.saveDataFrame(testWithPred, "test_result.csv")

    print("Seen {}".format(seen))
    print("Unseen {}".format(unseen))

    return testWithPred.loc[:, 'prediction']
