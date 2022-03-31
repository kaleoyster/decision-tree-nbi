"""--------------------------------------------
Descriptions:
    Data preprocessing file for decision tree

Author: Akshay Kale
Date: May 11th, 2021

TODO:
    1. Create Folders for the ouput [Done]
    2. Create Random forest model
    3. Complexity Parameters: Shapley
    4. Select the important variables [Done]
    5. Characterization of the clusters [Done]
    6. Computing deterioration scores,
        and intervention [Done]
    7. Implement Recursive feature elimination
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
from maps import *

def read_csv_file(path):
    """
    Description:
    """
    df = pd.read_csv(path, index_col=None)
    print(df.featureId.unique())
    return df


def main():
    # Read csvfile
    #path = 'nebraska_deepOutputsTICR/No Substructure - No Deck - YesSuperstructure.txt'
    path = 'nebraska_deepOutputsTICR/path.csv'
    read_csv_file(path)


if __name__=='__main__':
    main()
