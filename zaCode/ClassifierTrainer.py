from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

def trainLogisticRegression(xTrain, yTrain):
    classifier = LogisticRegression(n_jobs=-1, verbose=1, solver="lbfgs")

    classifier.fit(xTrain, yTrain)

    return classifier


def trainRandomForestClassifier(xTrain, yTrain):

    # 10000/3000 =>  {'n_estimators': 90 'max_features': 0.8, 'max_depth': 9}
    # classifier = RandomForestClassifier(n_estimators=90,max_features=0.8, max_depth=9, n_jobs=-1, verbose=1)

    # paramGrid = {
    #     "n_estimators":[80,90,100],
    #     "max_features":[0.7,0.8,0.9]
    # }


    #Best  choice is: {'max_features': 0.8, 'n_estimators': 100}
    classifier = RandomForestClassifier(n_jobs=-1, verbose=1, max_features=0.8, n_estimators=100)

    classifier.fit(xTrain, yTrain)


    return classifier



def trainGradientBoostingClassifier(xTrain, yTrain):

    # n_estimators = 120, learning_rate = 0.07
    # max_features= 0.5, max_depth= 6
    # subsample = 0.9
    classifier = GradientBoostingClassifier(n_estimators=120,max_depth=6,min_samples_leaf=1,learning_rate=0.07,max_features=0.5, verbose=1)
    # classifier = GradientBoostingClassifier(verbose=1)

    classifier.fit(xTrain, yTrain)

    return classifier



def trainNB(xTrain, yTrain):

    classifier = MultinomialNB()

    classifier.fit(xTrain, yTrain)

    return classifier


def trainUsingGridSearch(classifier, paramGrid, xTrain, yTrain):

    cv = StratifiedKFold(yTrain,n_folds=3)

    newClassifier = GridSearchCV(classifier, scoring="accuracy", param_grid=paramGrid, cv=cv, n_jobs=-1, verbose=1)

    newClassifier.fit(xTrain, yTrain)

    print("Best choice is: {}".format(newClassifier.best_params_))

    return newClassifier


def trainClassifier(xTrain,yTrain):

    print("Training classifier...")

    classifier = trainLogisticRegression(xTrain, yTrain)
    # classifier = trainGradientBoostingClassifier(xTrain, yTrain)
    # classifier = trainRandomForestClassifier(xTrain, yTrain)
    # classifier = trainNB(xTrain, yTrain)

    return classifier
