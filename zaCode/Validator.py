from sklearn import metrics
from sklearn.metrics import brier_score_loss

def performValidation(yPred, yTest):
    print(metrics.classification_report(yPred, yTest))

    print("\nNumber of test entries: {}".format(len(yTest)))

    print("\nScore: {}".format(sum(abs(yPred - yTest))))

    print("\nSquared error: {}".format( squared_err(yPred, yTest)))

def squared_err(yPred, yTest):
    return brier_score_loss(yTest, yPred)