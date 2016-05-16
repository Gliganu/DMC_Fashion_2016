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
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


def trainLogisticRegression(xTrain, yTrain):
    classifier = LogisticRegression(C=35, n_jobs=-1, verbose=1, solver="liblinear", class_weight='balanced')

    classifier.fit(xTrain, yTrain)

    return classifier


def trainRandomForestClassifier(xTrain, yTrain):
    # 10000/3000 =>  {'n_estimators': 90 'max_features': 0.8, 'max_depth': 9}
    classifier = RandomForestClassifier(n_estimators=90,max_features=0.8, max_depth=9, n_jobs=-1, verbose=1)
    #
    # paramGrid = {
    #    "n_estimators":[80,90,100],
    #    "max_features":[0.7,0.8,0.9]
    # }

    classifier = trainUsingGridSearch(classifier,paramGrid,xTrain,yTrain);

    # Best  choice is: {'max_features': 0.8, 'n_estimators': 100}
    classifier = RandomForestClassifier(n_jobs=-1, verbose=1, max_features=0.8, n_estimators=100)

    # classifier.fit(xTrain, yTrain)

    return classifier



def trainGradientBoostingClassifier(xTrain, yTrain):

    n_estimators = 150
    learning_rate = 0.05
    max_depth = 4
    max_features = 0.3
    min_samples_leaf = 5

    # max_features= 0.5, max_depth= 6
    # subsample = 0.9
    #  max_depth = 6, min_samples_leaf = 1, max_features = 0.5


    # classifier = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate,max_depth = 6, min_samples_leaf = 1, max_features = 0.5, verbose=1)
    classifier = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth,max_features=max_features, min_samples_leaf=min_samples_leaf, verbose=1)


    # paramGrid = {
    #     "min_samples_leaf":[1,3,5,7],
    # }
    # classifier = trainUsingGridSearch(classifier,paramGrid,xTrain,yTrain)


    classifier.fit(xTrain, yTrain)

    return classifier



def trainNB(xTrain, yTrain):

    classifier = MultinomialNB()
    # classifier = BernoulliNB()

    classifier.fit(xTrain, yTrain)

    return classifier


def trainUsingGridSearch(classifier, paramGrid, xTrain, yTrain):

    cv = StratifiedKFold(yTrain,n_folds=3)

    newClassifier = GridSearchCV(classifier, scoring="f1", param_grid=paramGrid, cv=cv, n_jobs=-1, verbose=1)

    newClassifier.fit(xTrain, yTrain)

    print("Best choice is: {}".format(newClassifier.best_params_))

    return newClassifier


def trainSVM(xTrain, yTrain, kernelType='rbf'):
    classifier = SVC(kernel=kernelType, verbose=True)
    classifier.fit(xTrain, yTrain)
    return classifier


def trainClassifier(xTrain, yTrain):
    print("Training classifier...")

    # classifier = trainLogisticRegression(xTrain, yTrain)
    classifier = trainGradientBoostingClassifier(xTrain, yTrain)
    # classifier = trainRandomForestClassifier(xTrain, yTrain)
    # classifier = trainNB(xTrain, yTrain)
    # classifier = trainSVM(xTrain, yTrain)

    # n_estimators = 150
    # learning_rate = 0.05
    # max_depth = 4
    # max_features = 0.3
    # min_samples_leaf = 5

    # paramGrid = {
    #   "n_estimators": [100, 150, 250, 400, 500],
    #   "learning_rate": [0.05, 0.1, 0.25, 0.5],
    #   "max_depth": [5, 10, 15, 20],
    #   "min_samples_leaf": [1, 5, 10]
    # }
    # classifier = trainUsingGridSearch(GradientBoostingClassifier(), paramGrid, xTrain, yTrain)

    return classifier
