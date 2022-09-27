"""-----------------------------------------------------------
descriptions:
    Data preprocessing file for decision tree

Author: Akshay Kale
Date: May 11th, 2021

TODO:
    1. Map for columns, make sure there is validity test for it.
--------------------------------------------------------------"""

# Data structures
from collections import defaultdict
import pandas as pd

# Model
#from mlxtend.frequent_patterns import apriori, association_rules
#from maps import *

def read_csv_file(path):
    """
    Description:
        Reads a csv file and returns a dataframe
    """
    _df = pd.read_csv(path, index_col=None)
    return _df

def create_map(list_of_columns):
    """
    Description:
        Return a column map
    """
    column_map = defaultdict()
    for index, column in enumerate(list_of_columns):
        #index = str(index + 1)
        column_map[index] = column
    return column_map

def main():
    """
    Driver function
    """
    # Read csvfile
    #path = 'nebraska_deepOutputsTICR/No Substructure - No Deck - YesSuperstructure.txt'
    path = 'nebraska_deepOutputsTICR/path.csv'
    _df = read_csv_file(path)
    for index, row in _df.iterrows():
        print(row)
    #print(_df)
    #column_map = create_map(list_of_columns)
    #df['featureName'] = df['featureId'].map(column_map)

if __name__=='__main__':
    main()
