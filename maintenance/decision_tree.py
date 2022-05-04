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

import sys
import csv

# Data structures
from collections import Counter
from collections import defaultdict
from decimal import Decimal
from numpy import array
import pandas as pd
import numpy as np
from tqdm import tqdm

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

# Model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Metrics and stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
#from sklearn.metrics import roc_auc_score

# Visualization
import plotly
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from maps import mapDict

def geo_coor_utility(longitude, latitude):
    """
    Function for converting longitude and latitude
    args:
        df: Dataframe
        columns: all the column

    return:
        df: dataframe with one hot encoding
    """
    if latitude > 0 and longitude > 0:
        lat = str(latitude)

        lat_degree = Decimal(lat[:2])
        print("printing degree:", lat_degree)
        lat_min = Decimal(lat[2:4])
        lat_min = (lat_min/60)
        print("printing min:", lat_min)
        lat_sec = Decimal(lat[4:8])
        lat_sec = (lat_sec/360000)
        print("printing sec:", lat_sec)
        lat_decimal = lat_degree + lat_min + lat_sec

        long = str(longitude)
        if len(long) <= 9:
            long  = long.zfill(9)
        long_degree = Decimal(long[:3])
        print("printing degree:", long_degree)
        long_min = Decimal(long[3:5])
        long_min = (long_min/60)
        print("printing min:", long_min)
        long_sec = Decimal(long[5:9])
        long_sec = (long_sec/360000)
        print("printing sec:", long_sec)
        long_decimal = - (long_degree + long_min + long_sec)
        print("-----------------------")
        long_decimal =  format(long_decimal, '.6f')
        lat_decimal =  format(lat_decimal, '.6f')
        return long_decimal, lat_decimal
    else:
        return 0.00, 0.00
    #except:
    #    return 0.00, 0.00

def convert_geo_coordinates(df, columns):
    """
    Function for converting longitude and latitude
    args:
        df: Dataframe
        columns: all the column

    return:
        df: dataframe with decimal longitude and latitude
    """
    longitudes = df['longitude']
    latitudes = df['latitude']

    transLongitudes = list()
    transLatitudes = list()

    for longitude, latitude in zip(longitudes, latitudes):
        tLongitude, tLatitude = geo_coor_utility(longitude,
                                                latitude)
        transLongitudes.append(tLongitude)
        transLatitudes.append(tLatitude)

    df['longitude'] = transLongitudes
    df['latitude'] = transLatitudes
    return df

def one_hot(df, columns):
    """
    Function for one-hot-encoding
    args:
        df: Dataframe
        columns: all the columns
        fields: specific fields to code one hot encoding

    return:
        columns: columns with one hot encoding
    """
    print("\n Printing columns, one hot encoding:")
    for column in columns:
        colMap = mapDict[column]
        data = df[column].map(colMap)
        values = array(data)

        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)

        # binary encoder
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        # define a new dictionary
        dictCol = defaultdict(list)
        for row in onehot_encoded:
            for index, value in zip(label_encoder.classes_, row):
                index = column + str(index)
                dictCol[index].append(value)
        for key in dictCol.keys():
            df[key] = dictCol[key]

        #TODO: Next, we have to figure out how do we scale these to other material
        # One-hot encoding categorial variable with high cardinality 
        # Cause inefficieny in tree-based ensembles. 
        # Continuous variables will be given more importance 
        # than the dummy variables by the algorithm 
        # which will obscure the order of feature 
        # importance resulting in poorer performance.
        # Further, we need to look at feature hasher and how it can help
        # Categorical variables: Material
    return df

# Function for normalizing
def normalize(df, columns):
    """
    Function for normalizing the data
    """
    print("Accessed the feature!")
    for feature in columns:
        print("printing the feature", feature)
        df[feature] = df[feature].astype(int)
        maxValue = df[feature].max()
        minValue = df[feature].min()
        df[feature] = (df[feature] - minValue) / (maxValue - minValue)
    return df

# Summarize features
def summarize_features(df, columns):
    """
    return df
    """
    for feature in columns:
        print("Feature :", feature)
        values = df[feature].astype(int)
        print_box_plot(values, filename=feature, col=feature)
        if feature == 'deteriorationScore':
            print("\n ",  Counter(df[feature]))

# Function for removing duplicates
def remove_duplicates(df, columnName='structureNumbers'):
    """
    Description: return a new df with duplicates removed
    Args:
        df (dataframe): the original dataframe to work with
        column (string): columname to drop duplicates by
    Returns:
        newdf (dataframe)
    """
    temp = list()
    for group in df.groupby(['structureNumber']):
        structureNumber, groupedDf = group
        groupedDf = groupedDf.drop_duplicates(subset=['structureNumber'],
                               keep='last'
                               )
        temp.append(groupedDf)
    newdf = pd.concat(temp)
    return newdf

def remove_null_values(df):
    """
    Description: return a new df with null values removed
    Args:
        df (dataframe): the original dataframe to work with
    Returns:
        df (dataframe): dataframe
    """
    for feature in df:
        if feature != 'structureNumber':
            try:
                df = df[~df[feature].isin([np.nan])]
            except:
                print("Error: ", feature)
    return df

def create_labels(df, label):
    """
    Description:
        Create binary categories from
        multiple categories.
    Args:
        df (dataframe)
        label (string): primary label
    Returns:
        df (dataframe): a dataframe with additional
        attributes
    """
    ## TODO: Create a new definition for positive class and negative class

    #positiveClass = df[df['cluster'].isin([label])]
    #negativeClass = df[~df['cluster'].isin([label, 'No intervention'])]
    print('Using this function')
    label = 'All intervention'
    label2 = 'No intervention'
    positiveClass = df[df['cluster'].isin([label])]
    negativeClass = df[df['cluster'].isin([label2])]

    positiveClass['label'] = ['positive']*len(positiveClass)
    negativeClass['label'] = ['negative']*len(negativeClass)
    df = pd.concat([positiveClass, negativeClass])
    return df

def categorize_attribute(df, fieldname, category=2):
    """
    Description:
        Categerize numerical values by normal distribution

    Args:
        df (Dataframe)
        category of attributes: Divide the attributes types
        either into a total number of 2 categories
        or 4 categories

    Returns:
       categories (list)
    """
    categories = list()
    mean = np.mean(df[fieldname])
    std = np.std(df[fieldname])
    if category == 4:
        for value in df[fieldname]:
            if value > mean and value < (mean + std):
                categories.append('Good')
            elif value > mean + std:
                categories.append('Good')
            elif value < (mean - std):
                categories.append('Bad')
            else:
                categories.append('Bad')
    elif category == 2:
        for value in df[fieldname]:
            if value > mean:
                categories.append("Good")
            else:
                categories.append("Bad")
    else:
        categories = list()

    return categories

# Confusion Matrix
def conf_matrix(cm, filename=''):
    """
    Description:
        Confusion matrix on validation set
    """
    indexList = list()
    columnList = list()
    filename = 'results/' + filename +  'ConfusionMatrix.png'

    for row in range(0, np.shape(cm)[0]):
        indexString = 'Actual' + str(row+1)
        columnString = 'Predicted' + str(row+1)
        indexList.append(indexString)
        columnList.append(columnString)

    dfCm = pd.DataFrame(cm,
                        index=indexList,
                        columns=columnList
                        )

    plt.subplots(figsize=(8, 8))
    sns.heatmap(dfCm, annot=True, fmt='g', cmap='BuPu')
    plt.savefig(filename)

# Box plot
def box_plot(scores, filename='', col='accuracy'):
    """
    Boxplot of training accuracy
    """
    filename = 'results/' + filename + 'AccuracyBoxPlot.png'
    df_score = pd.Series(scores)

    font = {'weight': 'bold',
            'size': 25
            }

    plt.figure(figsize=(10, 10))
    plt.title("Performance: Accuracy", **font)
    sns.boxplot(y=df_score, orient='v')
    plt.savefig(filename)

# Line plot
def line_plot(scores, filename=''):
    """
    lineplot of training accuracy
    """
    filename = "results/" + filename + 'AccuracyLinePlot.png'
    df_score = pd.Series(scores)

    font = {'weight': 'bold',
            'size': 25
            }

    plt.figure(figsize=(10, 10))
    plt.title("Performance: Accuracy", **font)
    sns.lineplot(data=df_score)
    plt.savefig(filename)

# Plot decision trees
def plot_decision_tree(model, filename=''):
    """
    Decision Tree
    """
    filename = "results/" + filename + "DecisionTree.png"
    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(model,
                   filled=True)
    fig.savefig(filename)

# Print splitnodes
def print_split_nodes(leaves, tree_structure, features):
    """
    Print the tree structure that includes
    leaves and split nodes.

    Collect the split nodes
    """

    # Unpack the tree stucture
    n_nodes, node_depth, children_left, children_right, feature, threshold = tree_structure

    # TODO:
    # Collect decision split nodes and convert them into csvfiles
    split_nodes = []

    # Create feature dictionary
    #feature_dict = {index:feat for index, feat in enumerate(features)}

    # Traverse the decision tree
    header = ['space',
             'node',
             'leftChild',
             'rightChild',
             'threshold',
             'feature']

    temp = []
    for i in range(n_nodes):
        if leaves[i]:
           #print("{space} node={node} is a leaf node and has"
           #      " the following tree structure:\n".format(
           #      space=node_depth[i]*"\t",
           #      node=i))
            temp.append(i)
        else:
            #print("{space} node is a split-node: "
            #      " go to node {left} if X[:, {feature}] <= {threshold} "
            #      " else to node {right}.".format(
            #     space=node_depth[i]*"\t",
            #     node=i,
            #     left=children_left[i],
            #     right=children_right[i],
            #     threshold=threshold[i],
            #     feature=feature_dict[feature[i]],
            #     ))
            split_nodes.append([node_depth[i],
                               i,
                               children_left[i],
                               children_right[i],
                               threshold[i],
                               feature])

    return header, split_nodes

# Navigate the decision tree
def find_leaves(e_best_model):
    """
    Navigate decision tree to find
    leaves
    """
    # Tree structure
    n_nodes = e_best_model.tree_.node_count
    children_left = e_best_model.tree_.children_left
    children_right = e_best_model.tree_.children_right
    feature = e_best_model.tree_.feature
    threshold = e_best_model.tree_.threshold

    # Initialize
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    leaves = np.zeros(shape=n_nodes, dtype=bool)

    # Start with the root node
    stack = [[0, 0]] # [[nodeId, depth]]
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to the 'stack'
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            leaves[node_id] = True

    tree_structure = (n_nodes,
                     node_depth,
                     children_left,
                     children_right,
                     feature,
                     threshold)

    return leaves, tree_structure

def print_decision_paths(clf, label, X_test, feature):
    """
    Description:
    Args:
    Returns:
    """
    n_nodes = clf.tree_.node_count
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_indicator = clf.decision_path(X_test)
    leaf_id = clf.apply(X_test)

    sample_id = 0
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]

    oStdout = sys.stdout
    fileName = label + '_' +'paths.txt'
    nodeList = []
    sampleIdList = []
    featureIdList = []
    valueList = []
    inequalityList = []
    thresholdList = []

    with open(fileName, 'w') as f:
        sys.stdout = f
        #print("Rules used to predict sample {id}:\n".format(id=sample_id))
        for record in X_test:
            for node_id in node_index:
                # continue to the next node if it is a leaf node
                if leaf_id[sample_id] == node_id:
                    continue
                # check if value of the split feature for sample 0 is below threshold
                if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"
               # print(
               #         "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
               #         "{inequality} {threshold})".format(
               #         node=node_id,
               #         sample=sample_id,
               #         feature=feature[node_id],
               #         value=X_test[sample_id, feature[node_id]],
               #         inequality=threshold_sign,
               #         threshold=threshold[node_id],
               #         )
               #     )
                    nodeList.append(node_id)
                    sampleIdList.append(sample_id)
                    featureIdList.append(feature[node_id])
                    valueList.append(X_test[sample_id, feature[node_id]])
                    inequalityList.append(threshold_sign)
                    thresholdList.append(threshold[node_id])
        sys.stdout = oStdout

        sample_ids = [0, 1, 2]
        # boolean array indicating the nodes both samples go through
        common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
        # obtain node ids using position in array
        common_node_id = np.arange(n_nodes)[common_nodes]

        #print(
        #"\nThe following samples {samples} share the node(s) {nodes} in the tree".format(
        #    samples=sample_ids,
        #    nodes=common_node_id
        #))
        #print("This is {prop}% of all nodes.".format(prop=100 * len(common_node_id) / n_nodes))
        data = pd.DataFrame({
                     'node': nodeList,
                     'sampleId': sampleIdList,
                     'featureId': featureIdList,
                     'valueId': valueList,
                     'inequality': inequalityList,
                     'threshold': thresholdList
                    })

        data.to_csv("path.csv")

# To summarize performance
def performance_summarizer(eKappaDict, gKappaDict,
                          eConfDict, gConfDict,
                          eClassDict, gClassDict,
                          eAccDict, gAccDict,
                          #eRocsDict, gRocsDict,
                          eModelsDict, gModelsDict,
                          eFeatureDict, gFeatureDict, testX, cols, label):

    """
    Description:
        Summarize the prformance of the decision
    Args:
        Kappa Values (list):
        Confusion Matrix (list):
        Accuracy Values (list):
        Models Values (list):
        Features Values (list):

    Returns:
        Prints a summary of Model performance with respect to
        Entropy and Kappa Value
    """
    # Entropy
    eBestKappa = max(eKappaDict.keys())
    eBestAcc = max(eAccDict.keys())
    eBestDepth = eKappaDict.get(eBestKappa)
    ecm = eConfDict.get(eBestDepth)
    efi = eFeatureDict.get(eBestDepth)
    efi = dict(sorted(efi.items(), key=lambda item: item[1]))

    print("""\n
            -------------- Performance of Entropy ---------------
            \n""")
    print("\n Best Kappa Values: ", eBestKappa)
    print("\n Best Accuracy: ", eBestAcc)
    print("\n Best Depth: ", eBestDepth)
    print("\n Classification Report: \n", eClassDict.get(eBestDepth))
    print("\n Confusion Matrix: \n", ecm)
    print("\n Feature Importance: \n", efi)
    #print("\n AUC: ", eRocsDict[eBestDept])

    # GiniIndex 
    gBestKappa = max(gKappaDict.keys())
    gBestAcc = max(gAccDict.keys())
    gBestDepth = gKappaDict.get(gBestKappa)
    gcm = gConfDict.get(gBestDepth)
    gfi = gFeatureDict.get(gBestDepth)
    gfi = dict(sorted(gfi.items(), key=lambda item: item[1]))

    print("""\n
             ----------- Performance with GiniIndex ------------
             \n""")

    print("\n Best Kappa Values: ", gBestKappa)
    print("\n Best Accuracy: ", gBestAcc)
    print("\n Best Depth: ", gBestDepth)
    print("\n Classification Report: \n", gClassDict.get(gBestDepth))
    print("\n Confusion Matrix: \n", gcm)
    print("\n Feature Importance: \n", efi)
    #print("\n AUC: ", gRocsDict[gBestDept])

    # Plot Confusion Matrix
    conf_matrix(gcm, 'Gini')
    conf_matrix(ecm, 'Entropy')

    # Box plot of training accuracies
    scoresGini = list(gAccDict.keys())
    scoresEntropy = list(eAccDict.keys())

    box_plot(scoresGini, 'Gini')
    box_plot(scoresEntropy, 'Entropy')

    ## Line plot
    line_plot(scoresGini, 'Gini')
    line_plot(scoresEntropy, 'Entropy')

    # Get best models (entropy and gini models)
    eBestModel = eModelsDict.get(eBestDepth)
    gBestModel = gModelsDict.get(gBestDepth)

    # Printing Node Counts
    # TODO: DELETE the create nNodes
    print("\nPrinting split-nodes")
    leaves, treeStructure = find_leaves(eBestModel)
    splitNodes = print_split_nodes(leaves, treeStructure, cols)
    print_decision_paths(eBestModel, label, testX, cols)

    # Entropy
    # TODO: Save a decision tree model for every run
            # This function can be perhaps saved somewhere else.

    # Print decision tree of the Best Model
    print("\n Saving decision trees \n")
    eTextRepresentation = tree.export_text(eBestModel)

    # models / the type of decision tree model / entropy
    saveLogFileEntropyName ="models/"+label+"_entropy_decision_tree.log"
    saveLogFileGiniName ="models/"+label
    "_gini_decision_tree.log"
    with open(saveLogFileEntropyName, "w") as fout:
        fout.write(eTextRepresentation)

    # Gini
    gTextRepresentation = tree.export_text(gBestModel)
    with open(saveLogFileGiniName, "w") as fout:
        fout.write(gTextRepresentation)

    print("\n Plotting decision trees \n")

    plot_decision_tree(eBestModel, filename='Entropy')
    plot_decision_tree(gBestModel, filename='Gini')

    #TODO: What is this dangling function?
    #with open("models/splitnodes.log", "w") as fout:
    #    fout.write(splitNodes)

    return (eBestKappa, gBestKappa), (eBestAcc, gBestAcc), (efi, gfi), (eBestModel, gBestModel)

def tree_utility(train_x, trainy,
                 test_x, testy, cols,
                 criteria='gini', max_depth=7):
    """
    Description:
        Performs the modeling and returns performance metrics

    Args:
        trainX: Features of Training Set
        trainy: Ground truth of Training Set
        testX: Features of Testing Set
        testy: Ground truth of Testing Set

    Return:
        acc: Accuracy
        cm: Confusion Report
        cr: Classification Report
        kappa: Kappa Value
        model: Decision Tree Model
    """
    model = DecisionTreeClassifier(criterion=criteria, max_depth=max_depth)
    #model = HistGradientBoostingClassifier(categorical_features=[], max_depth=maxDepth)
    model.fit(train_x, trainy)
    prediction = model.predict(test_x)
    acc = accuracy_score(testy, prediction)
    _cm = confusion_matrix(testy, prediction)
    _cr = classification_report(testy, prediction, zero_division=0)
    _fi = dict(zip(cols, model.feature_importances_))
    #rocAuc = roc_auc_score(testy, prediction, multi_class='ovr')
    kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    return acc, _cm, _cr, kappa, model, _fi # rocAuc, model

# Decision Tree
def decision_tree(X, y, features, label, nFold=5):
    """
    Description:
        Performs training-testing split
        Train model for various depth level
        Train model for both Entropy and GiniIndex

    Args:
        df (Dataframe)
    """
    # Kfold cross validation
    kfold = KFold(nFold, shuffle=True, random_state=1)

    # For storing Confusion Matrix
    confusionMatrixsEntropy = []
    confusionMatrixsGini = []

    # For storing Classification Report
    classReportsEntropy = []
    classReportsGini = []

    # Scores
    scoresGini = []
    scoresEntropy = []

    # ROC AUC 
    eRocs = []
    gRocs = []

    # Kappa values
    gKappaValues = []
    eKappaValues = []

    # Converting them to array
    cols = X.columns
    X = np.array(X)
    y = np.array(y)

    # Store models:
    eModels = []
    gModels = []

    # Feature importance
    eFeatures = []
    gFeatures = []

    for depth in tqdm(range(1, 31), desc='\n Modeling DT'):
        tempG = []
        tempE = []
        for foldTrainX, foldTestX in kfold.split(X):
            trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                          X[foldTestX], y[foldTestX]

            # Gini
            gacc, gcm, gcr, gkappa, gmodel, gfi = tree_utility(trainX, trainy,
                                                 testX, testy, cols,
                                                 criteria='gini',
                                                 max_depth=depth
                                                 )

            # Entropy
            eacc, ecm, ecr, ekappa, emodel, efi = tree_utility(trainX, trainy,
                                                  testX, testy, cols,
                                                  criteria='entropy',
                                                  max_depth=depth )
            tempG.append(gacc)
            tempE.append(eacc)

        # Accuracies
        scoresGini.append(np.mean(tempG))
        scoresEntropy.append(np.mean(tempE))

        # Confusion Matrix
        confusionMatrixsEntropy.append(ecm)
        confusionMatrixsGini.append(gcm)

        # Classification Report
        classReportsEntropy.append(ecr)
        classReportsGini.append(gcr)

        # Kappa Values (TODO: select average of Kappa Value)
        eKappaValues.append(ekappa)
        gKappaValues.append(gkappa)

        # ROC AUC values(TODO: select average of Kappa Value)
        #eRocs.append(eroc)
        #gRocs.append(groc)

        # Models
        eModels.append(emodel)
        gModels.append(gmodel)

        # Feature importance
        eFeatures.append(efi)
        gFeatures.append(gfi)

    # Performance Summarizer
    depths = list(range(1, 31))

    # Create Dictionaries
    # Kappa
    eKappaDict = dict(zip(eKappaValues, depths))
    gKappaDict = dict(zip(gKappaValues, depths))

    # Confusion Matrix 
    eConfDict = dict(zip(depths, confusionMatrixsEntropy))
    gConfDict = dict(zip(depths, confusionMatrixsGini))

    # Classification Report
    eClassDict = dict(zip(depths, classReportsEntropy))
    gClassDict = dict(zip(depths, classReportsGini))

    # Scores (Accuracy)
    eScoreDict = dict(zip(scoresEntropy, depths))
    gScoreDict = dict(zip(scoresGini, depths))

    #TODO:  Scores (ROCs) doesn't work
    #eRocsDict = dict(zip(eRocs, depths))
    #gRocsDict = dict(zip(gRocs, depths))

    # Models
    eModelsDict = dict(zip(depths, eModels))
    gModelsDict = dict(zip(depths, gModels))

    # Feature Importance
    eFeatureDict = dict(zip(depths, eFeatures))
    gFeatureDict = dict(zip(depths, gFeatures))

    kappaVals, accVals, featImps, models = performance_summarizer(eKappaDict, gKappaDict,
                                           eConfDict, gConfDict,
                                           eClassDict, gClassDict,
                                           eScoreDict, gScoreDict,
                                           #eRocsDict, gRocsDict,
                                           eModelsDict, gModelsDict,
                                           eFeatureDict, gFeatureDict, testX, cols, label)

    # Return the average kappa value for state
    eBestModel, gBestModel = models
    #leaves = find_leaves(eBestModel)
    #splitNodes = print_split_nodes(leaves, eBestModel, features)
    return kappaVals, accVals, featImps, models

def plot_centroids(states, centroid_df, metric_name):
    """
    Description:

    Args:
        states (states)
        listOfMetricsValues (list of list)
        metricName (list)

    Returns:
        saves a 3d scatter plot
    """
    filename = metric_name + ".html"
    title = "3D representation of centroids for the midwestern states"
    fig = px.scatter_3d(centroid_df,
                      x='subNumInt',
                      y='supNumInt',
                      z='deckNumInt',
                      color='name',
                      symbol='state',
                      title=title)

    plotly.offline.plot(fig, filename=filename)


def plot_overall_performance(states, list_of_metric_values, metric_name, state):
    """
    Description:
        plots a barchart of all states and their metrics values
    Args:
        states (list)
        listOfMetricValues (list of list)
        metricName (list of names)
    Returns:
        saves a barchart
    """
    filename = metric_name + '.png'

    # Values
    e_metric_values = []
    g_metric_values = []
    for metric_val in list_of_metric_values:
        e_metric, g_metric = metric_val
        e_metric_values.append(e_metric)
        g_metric_values.append(g_metric)

    height = np.array(range(0, len(states)))

    # Make the plot
    plt.figure(figsize=(10, 8))
    plt.title("Overall Performance")
    plt.bar(height, e_metric_values,
            color='#7f6d5f', width=0.25, label='gini')
    plt.bar(height + 0.25,
            g_metric_values, color='#557f2d',
            width=0.25, label='entropy')
    plt.xticks(height, states, rotation=45)
    plt.legend()
    plt.savefig(filename)

    print("\n" + metricName + "Table: ")
    dataFrame = pd.DataFrame()
    dataFrame['state'] = states
    dataFrame['gini'] = gMetricValues
    dataFrame['entropy'] = eMetricValues
    print(dataFrame)

def to_csv(list_of_dataframes):
    """
    Description:
        Convert the dataframe into csv files
    Args:
        listOfDataFrames: (list of dataframe)
    """
    concat_df = pd.concat(list_of_dataframes)
    concat_df.to_csv('allFiles.csv', sep=',', index=False)

def read_csv(csv_file):
    """
    reads csvfile using pandas
    """
    _df = pd.read_csv(csv_file)
    return _df

def read_geo_coordinates(_df):
    """
    return latitude and longitude
    """
    latitude = _df['latitude']
    longitude = _df['longitude']
    geo_coordinates = list(zip(latitude, longitude))
    return geo_coordinates

def plot_histogram(list_longitude, list_latitude):
    """
    return a historgram
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(
    x=list_longitude,
    histnorm='percent',
    name='control', # name used in legend and hover labels
    xbins=dict( # bins used for histogram
        start=-4.0,
        end=3.0,
        size=0.5
    ),
    marker_color='#EB89B5',
    opacity=0.75
    ))

    fig.add_trace(go.Histogram(
    x=list_latitude,
    histnorm='percent',
    name='experimental',
    xbins=dict(
        start=-3.0,
        end=4,
        size=0.5
    ),
    marker_color='#330C73',
    opacity=0.75
    ))

    fig.update_layout(
    title_text='Sampled Results', # title of plot
    xaxis_title_text='Value', # xaxis label
    yaxis_title_text='Count', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
    )

    fig.show()

#def main():
#    """
#    Driving function
#    """
#    #latitude, longitude = '42283880', '10252195'
#    #longitude, latitude = '081055065', '35271855'
#    #converted_long, converted_lat = geo_coor_utility(longitude, latitude)
#    #print(converted_lat,',',converted_long)
#    csvfile = "nebraska_deep.csv"
#    _df = read_csv(csvfile)
#    geo_coordinates = read_geo_coordinates(_df)
#    list_latitude = []
#    list_longitude = []
#
#    conv_latitude = []
#    conv_longitude = []
#    for coor in geo_coordinates:
#        latitude, longitude = coor
#        list_longitude.append(longitude)
#        list_latitude.append(latitude)
#        converted_long, converted_lat = geo_coor_utility(longitude, latitude)
#        print(converted_lat,', ',converted_long)
#        conv_longitude.append(converted_long)
#        conv_latitude.append(converted_lat)
#    plot_histogram(list_longitude, list_latitude)
#    plot_histogram(conv_longitude, conv_latitude)


#main()
