import FileManager
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import seaborn as sns



def plotReturnQuantityHist():

    data = FileManager.getWholeTrainingData()

    sns.distplot(data['returnQuantity'],kde=False)

    plt.title("ReturnQuantity Histogram")
    plt.show()


def plotScatterPlots():

    data = FileManager.getWholeTrainingData()

    sns.jointplot(data['returnQuantity'],data['price'])

    plt.title("ReturnQuantity x Price Scatterplot")
    plt.show()


def plotPairPlot():

    data = FileManager.getWholeTrainingData()

    # data = data[['colorCode','sizeCode','quantity','price','rrp','voucherAmount','returnQuantity']]
    data = data[['colorCode','returnQuantity']]

    sns.pairplot(data)

    plt.title("Pairwise comparison of features")
    plt.show()




