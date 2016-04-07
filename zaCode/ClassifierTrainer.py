from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


def trainLogisticRegression(xTrain, yTrain):
    classifier = LogisticRegression(n_jobs=-1, verbose=1)

    classifier.fit(xTrain, yTrain)

    return classifier


def trainRandomForestClassifier(xTrain, yTrain):

    # 10000/3000 =>  {'n_estimators': 90 'max_features': 0.8, 'max_depth': 9}
    # classifier = RandomForestClassifier(n_estimators=90,max_features=0.8, max_depth=9, n_jobs=-1, verbose=1)

    classifier = RandomForestClassifier(n_jobs=-1, verbose=1)

    classifier.fit(xTrain, yTrain)

    return classifier



def trainGradientBoostingClassifier(xTrain, yTrain):

    # n_estimators = 120, learning_rate = 0.07
    # max_features= 0.5, max_depth= 6
    # subsample = 0.9
    # classifier = GradientBoostingClassifier(n_estimators=120,max_depth=6,min_samples_leaf=1,learning_rate=0.07,max_features=0.5, verbose=1)
    classifier = GradientBoostingClassifier(verbose=1)

    classifier.fit(xTrain, yTrain)

    return classifier



def trainClassifier(xTrain,yTrain):

    print "Training classifier..."

    # classifier = trainLogisticRegression(xTrain, yTrain)
    classifier = trainGradientBoostingClassifier(xTrain, yTrain)
    # classifier = trainRandomForestClassifier(xTrain, yTrain)

    return classifier
