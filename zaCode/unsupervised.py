import math
import sys

from collections import defaultdict
from copy import copy

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

from zaCode.DatasetManipulator import DSetTransform

class ProbEngine:
    """ Performs Probability manipulations on a dataset
    """
    
    def __init__(self, data):
        self.data = data
        
    def histogram(self, col, categorical = True):
        """ computes histogram on column given 
            histogram returned in a dict vals -> counts for 
        """
        ret = { l: 0 for l in data[col].unique() }
        for idx, val in self.data[col].iteritems():
            ret[vat] += 1
        return ret
        
        
    def kde(self, col):
        """ computes kde smothing of histogram on column given """
        # TODO implement
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        