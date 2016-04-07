from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


import matplotlib.pyplot as plt
import seaborn as sns

def performValidation(yPred, yTest):
    print metrics.classification_report(yPred, yTest)

    numberOver = sum(yPred > yTest)
    numberUnder = sum(yPred < yTest)
    numberEqual = sum(yPred == yTest)

    print "Number Over {}".format(numberOver)
    print "Number Under {}".format(numberUnder)
    print "Number Equal {}".format(numberEqual)

    print  sum(abs(yPred - yTest))