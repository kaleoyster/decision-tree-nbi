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

def geo_coor_utility(longitude, latitude):
    """
    Function for converting longitude and latitude
    args:
        df: Dataframe
        columns: all the column

    return:
        df: dataframe with one hot encoding
    """
    try:
         lat = latitude
         latDegree = int(lat[:2])
         latMin = int(lat[2:4])
         latMin = (latMin/60)
         latSec = int(lat[4:8])
         latSec = (latSec/360000)
         latDecimal = latDegree + latMin + latSec
         long = longitude
         longDegree = int(long[:3])
         longMin = int(long[3:5])
         longMin = (longMin/60)
         longSec = int(long[5:9])
         longSec = (longSec/360000)
         longDecimal = -(longDegree + longMin + longSec)
         return longDecimal, latDecimal
    except:
        return 0.00, 0.00

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

def oneHot(df, columns):
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

        #TODO: Next, we have to figure out how do we scale these to other makerials
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
    for feature in columns:
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
    dfScore = pd.Series(scores)

    font = {'weight': 'bold',
            'size': 25
            }

    plt.figure(figsize=(10, 10))
    plt.title("Performance: Accuracy", **font)
    sns.boxplot(y=dfScore, orient='v')
    plt.savefig(filename)

# Line plot
def line_plot(scores, filename='', col='accuracy'):
    """
    lineplot of training accuracy
    """
    filename = "results/" + filename + 'AccuracyLinePlot.png'
    dfScore = pd.Series(scores)

    font = {'weight': 'bold',
            'size': 25
            }
    plt.figure(figsize=(10, 10))
    plt.title("Performance: Accuracy", **font)
    sns.lineplot(data=dfScore)
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
def print_split_nodes(leaves, treeStructure, features):
    """
    Print the tree structure that includes
    leaves and split nodes.

    Collect the split nodes
    """

    # Unpack the tree stucture
    nNodes, nodeDepth, childrenLeft, childrenRight, feature, threshold = treeStructure

    # TODO:
    # Collect decision split nodes and convert them into csvfiles
    splitNodes = list()

    # Create feature dictionary
    featureDict = {index:feat for index, feat in enumerate(features)}

    # Traverse the decision tree
    header = ['space',
             'node',
             'leftChild',
             'rightChild',
             'threshold',
             'feature']

    temp = list()
    for i in range(nNodes):
        if leaves[i]:
           #print("{space} node={node} is a leaf node and has"
           #      " the following tree structure:\n".format(
           #      space=nodeDepth[i]*"\t",
           #      node=i))
           temp.append(i)
        else:
            #print("{space} node is a split-node: "
            #      " go to node {left} if X[:, {feature}] <= {threshold} "
            #      " else to node {right}.".format(
            #     space=nodeDepth[i]*"\t",
            #     node=i,
            #     left=childrenLeft[i],
            #     right=childrenRight[i],
            #     threshold=threshold[i],
            #     feature=featureDict[feature[i]],
            #     ))

            splitNodes.append([nodeDepth[i],
                               i,
                               childrenLeft[i],
                               childrenRight[i],
                               threshold[i],
                               feature])

    return header, splitNodes

# Navigate the decision tree
def find_leaves(eBestModel):
    """
    Navigate decision tree to find
    leaves
    """

    # Tree structure
    nNodes = eBestModel.tree_.node_count
    childrenLeft = eBestModel.tree_.children_left
    childrenRight = eBestModel.tree_.children_right
    feature = eBestModel.tree_.feature
    threshold = eBestModel.tree_.threshold

    # Initialize
    nodeDepth = np.zeros(shape=nNodes, dtype=np.int64)
    leaves = np.zeros(shape=nNodes, dtype=bool)

    # Start with the root node
    stack = [[0, 0]] # [[nodeId, depth]]
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        nodeId, depth = stack.pop()
        nodeDepth[nodeId] = depth

        # If the left and right child of a node is not the same we have a split
        isSplitNode = childrenLeft[nodeId] != childrenRight[nodeId]
        # If a split node, append left and right children and depth to the 'stack'
        if isSplitNode:
            stack.append((childrenLeft[nodeId], depth + 1))
            stack.append((childrenRight[nodeId], depth + 1))
        else:
            leaves[nodeId] = True

    treeStructure = (nNodes,
                     nodeDepth,
                     childrenLeft,
                     childrenRight,
                     feature,
                     threshold)

    return leaves, treeStructure

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
    nodeList = list()
    sampleIdList = list()
    featureIdList = list()
    valueList = list()
    inequalityList = list()
    thresholdList = list()

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

def tree_utility(trainX, trainy,
                 testX, testy, cols,
                 criteria='gini', maxDepth=7):
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
    model = DecisionTreeClassifier(criterion=criteria, max_depth=maxDepth)
    #model = HistGradientBoostingClassifier(categorical_features=[], max_depth=maxDepth)
    model.fit(trainX, trainy)
    prediction = model.predict(testX)
    acc = accuracy_score(testy, prediction)
    cm = confusion_matrix(testy, prediction)
    cr = classification_report(testy, prediction, zero_division=0)
    fi = dict(zip(cols, model.feature_importances_))
    #rocAuc = roc_auc_score(testy, prediction, multi_class='ovr')
    kappa = cohen_kappa_score(prediction, testy,
                              weights='quadratic')
    return acc, cm, cr, kappa, model, fi # rocAuc, model

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
    confusionMatrixsEntropy = list()
    confusionMatrixsGini = list()

    # For storing Classification Report
    classReportsEntropy = list()
    classReportsGini = list()

    # Scores
    scoresGini = list()
    scoresEntropy = list()

    # ROC AUC 
    eRocs = list()
    gRocs = list()

    # Kappa values
    gKappaValues = list()
    eKappaValues = list()

    # Converting them to array
    cols = X.columns
    X = np.array(X)
    y = np.array(y)

    # Store models:
    eModels = list()
    gModels = list()

    # Feature importance
    eFeatures = list()
    gFeatures = list()

    for depth in tqdm(range(1, 31), desc='\n Modeling DT'):
        tempG = list()
        tempE = list()
        for foldTrainX, foldTestX in kfold.split(X):
            trainX, trainy, testX, testy = X[foldTrainX], y[foldTrainX], \
                                          X[foldTestX], y[foldTestX]

            # Gini
            gacc, gcm, gcr, gkappa, gmodel, gfi = tree_utility(trainX, trainy,
                                                 testX, testy, cols,
                                                 criteria='gini',
                                                 maxDepth=depth
                                                 )

            # Entropy
            eacc, ecm, ecr, ekappa, emodel, efi = tree_utility(trainX, trainy,
                                                  testX, testy, cols,
                                                  criteria='entropy',
                                                  maxDepth=depth )
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

def plot_centroids(states, centroidDf, metricName):
    """
    Description:

    Args:
        states (states)
        listOfMetricsValues (list of list)
        metricName (list)

    Returns:
        saves a 3d scatter plot
    """
    filename = metricName + ".html"
    title = "3D representation of centroids for the midwestern states"

    fig = px.scatter_3d(centroidDf,
                      x='subNumInt',
                      y='supNumInt',
                      z='deckNumInt',
                      color='name',
                      symbol='state',
                      title=title)

    plotly.offline.plot(fig, filename=filename)


def plot_overall_performance(states, listOfMetricValues, metricName, state):
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
    filename = metricName + '.png'

    # Values
    eMetricValues = list()
    gMetricValues = list()
    for metricVal in listOfMetricValues:
        eMetric, gMetric = metricVal
        eMetricValues.append(eMetric)
        gMetricValues.append(gMetric)

    height = np.array(range(0, len(states)))

    # Make the plot
    plt.figure(figsize=(10, 8))
    plt.title("Overall Performance")
    plt.bar(height, eMetricValues, color='#7f6d5f', width=0.25, label='gini')
    plt.bar(height + 0.25, gMetricValues,color='#557f2d', width=0.25, label='entropy')
    plt.xticks(height, states, rotation=45)
    plt.legend()
    plt.savefig(filename)

    print("\n" + metricName + "Table: ")
    dataFrame = pd.DataFrame()
    dataFrame['state'] = states
    dataFrame['gini'] = gMetricValues
    dataFrame['entropy'] = eMetricValues
    print(dataFrame)

def to_csv(listOfDataFrames):
    """
    Description:
        Convert the dataframe into csv files
    Args:
        listOfDataFrames: (list of dataframe)
    """
    concatDf = pd.concat(listOfDataFrames)
    concatDf.to_csv('allFiles.csv', sep=',', index=False)
