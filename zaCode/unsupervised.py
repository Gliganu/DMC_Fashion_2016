import math
import sys

from collections import defaultdict
from copy import copy

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.covariance import EmpiricalCovariance, MinCovDet

from zaCode.DatasetManipulator import DSetTransform


class ProbEngine:
    """ Performs Probability manipulations on a data set
    """
    
    def __init__(self, data):
        self.data = data
        self.robustCov_ = None
        self.globalCov_ = None

    def histogram(self, col, categorical = True):
        """ computes histogram on column given 
            histogram returned in a dict vals -> counts for 
        """
        ret = { l: 0 for l in self.data[col].unique() }
        for idx, val in self.data[col].iteritems():
            ret[val] += 1
        return ret

    def robust_cov(self):
        """
        Gives a robust estimate of covariance matrix of given data set.
        (i.e. removes outliers from data set to minimise cov mat determinant)
        :return: features x features covariance matrix
        """
        if self.robustCov_ is None:
            self.robustCov_ = MinCovDet().fit(self.data)

        return self.robustCov_

    def global_cov(self):
        """
        Gives a global maximum likelihood estimate of the covariance matrix of the data set)
        (no outliers removed)
        :return: features x features covariance matrix
        """
        if self.globalCov_ is None:
            self.globalCov_ = EmpiricalCovariance().fit(self.data)

        return self.globalCov_

    def robust_mh_dist(self, row):
        """
        return mahalanobis distance based on robust fitted covariance matrix and mean
        :param row: numpy array-like row of features
        :return: distance (double)
        """
        cv = self.robust_cov()
        return cv.mahalanobis(row)

    def global_mh_dist(self, row):
        """
        returns mahalanobis distance based on global fitter covariance matrix mean
        :param row: numpy array-like row of features
        :return: distance (double)
        """
        cv = self.global_cov()
        return cv.mahalanobis(row)

    def add_mh_feats(self, other_data):
        """
        adds mahalanobis distance to robust and global cvmat, mean params estimated
        from target dataset to other_data
        :param other_data: data set to add features to
        :return: other_data, with additionat features (other_data is mutated)
        """
        print("estimating covariances...")

        rb_cv = self.robust_cov()

        print("robust estimate done, continuing with global est...")

        gb_cv = self.global_cov()

        print("global estimate done.")
        print("data to fill in with features has {} rows".format(len(other_data)))
        print("starting to add features...")

        cnt = 0
        for idx, vals in other_data.iterrows():
            other_data.loc[idx, "robust_MhlDist"] = rb_cv.mahalanobis(vals)
            other_data.loc[idx, "global_MhlDist"] = gb_cv.mahalanobis(vals)

            cnt += 1
            if cnt % 20000 == 0:
                print("just passed row {}...".format(cnt))

        return other_data

        
        
        
        
        






























        