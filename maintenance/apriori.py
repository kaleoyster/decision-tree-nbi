"""--------------------------------------------
Descriptions:
    Data preprocessing file for decision tree

Author: Akshay Kale
Date: May 11th, 2021

TODO:
    1. Map for columns, make sure there is validity test for it.
-----------------------------------------------"""

# Data structures
import sys
import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from collections import Counter
from collections import defaultdict
from tqdm import tqdm

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Model
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from mlxtend.frequent_patterns import apriori, association_rules

# Metrics and stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score

# Visualization
import plotly
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from maps import *

def read_csv_file(path):
    """
    Description:
    """
    df = pd.read_csv(path, index_col=None)
    return df

def create_map(listOfColumns):
    """
    Description:

    """
    columnMap = defaultdict()
    for index, column in enumerate(listOfColumns):
        #index = str(index + 1)
        columnMap[index] = column
    return columnMap

def main():
    # Read csvfile
    #path = 'nebraska_deepOutputsTICR/No Substructure - No Deck - YesSuperstructure.txt'
    path = 'nebraska_deepOutputsTICR/path.csv'
    df = read_csv_file(path)
    columnMap = create_map(listOfColumns)
    df['featureName'] = df['featureId'].map(columnMap)
    print(df['featureName'].unique())


if __name__=='__main__':
    main()
