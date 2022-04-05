"""---------------------------------------------------------->
Description: preprocessing library
Author: Akshay Kale
Date: May 7th, 2021
<----------------------------------------------------------"""

# Data structures
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict
from tqdm import tqdm

# Metrics and stats
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
import scipy.cluster.hierarchy as shc
from sklearn.cluster import KMeans
from kneed import KneeLocator

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly
import plotly.express as px

# Function for normalizing
def normalize(df, columns):
    """
    Description:
        Function for normalizing the data
    """
    for feature in columns:
        df[feature] = df[feature].astype(int)
        maxValue = df[feature].max()
        minValue = df[feature].min()
        df[feature] = (df[feature] - minValue) / (maxValue - minValue)
    return df

# Function to group by
def drop_duplicate_rows(df):
    """
    Description:
        Function to groupby bridge records
    """
    df = df.drop_duplicates(subset=['structureNumber'],
                                keep='first',
                                inplace=True)
    return df

# Differences in between the clusters 
def ANOVA(df, feature):
    """
    Description:
        Takes in dataframe with three columns, each column
        presents the value of an attribute from each cluster.

    Args:
        df (dataframe)
        feature (list)

    Returns:
        anovaTable (dataframe)
    """
    dfMelt = pd.melt(df.reset_index(), id_vars=['index'])
    dfMelt.columns = ['index', 'cluster', 'value']
    plot_box(dfMelt)
    model = ols('value ~ C(cluster)',
                data=dfMelt).fit()
    anovaTable = sm.stats.anova_lm(model,
                                   typ=2)
    tukey = pairwise_tukeyhsd(endog=dfMelt['value'],
                             groups=dfMelt['cluster'],
                             alpha=0.05)
    return anovaTable, tukey

def plot_elbow(sse):
    """
    Description:
        Plot the elbow to find the optimal number of
    clusters
    """
    filename = "results/" + "OptimalClustersElbow.png"
    plt.figure(figsize=(10, 10))
    plt.style.use("fivethirtyeight")
    plt.title("Optimal Number of Clusters using Elbow Method")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.savefig(filename)

def plot_silhouette(silhouette):
    """
    Description:
        Plot silhouette constant for finding optimal number
    of clusters
    """
    filename = "results/" + "Silhouette.png"
    plt.style.use("fivethirtyeight")
    plt.title("Plotting Silhouette")
    plt.plot(range(2, 11), silhouette)
    plt.xticks(range(2, 12))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette coefficient")
    plt.savefig(filename)

def plot_scatter(col1, col2, color=None):
    """
    Description:
        Scatter plot for maximum number of bridges
    """
    filename = "results/" + "ScatterPlot.png"
    plt.figure(figsize=(10, 10))
    plt.style.use("fivethirtyeight")
    plt.title("Baseline Difference Score Vs. Deterioration Score ")
    plt.scatter(col2, col1)
    plt.xlabel("BaselineDiferenceScore")
    plt.ylabel("DeteriorationScore")
    plt.savefig(filename)

def plot_scatter_Kmeans(df):
    """
    Description:
        Scatter plot for displaying k-means clusters
    """
    filename = "results/" +  "kmeans.png"
    cluster1 = df[df['cluster'] == 0]
    cluster2 = df[df['cluster'] == 1]
    cluster3 = df[df['cluster'] == 2]

    plt.figure(figsize=(10, 10))
    plt.style.use("fivethirtyeight")
    plt.title("Plotting Clusters")

    plt.scatter(cluster1['deck'],
                cluster1['superstructure'],
                color='red')

    plt.scatter(cluster2['deck'],
                cluster2['superstructure'],
                color='green')

    plt.scatter(cluster3['deck'],
                cluster3['superstructure'],
                color='blue')

    plt.xlabel("Column 1")
    plt.ylabel("Column 2")
    plt.savefig(filename)

# Plot boxplot
def plot_box(dfMelt, name='', x='cluster', y='value'):
    """
    Description:
        Scatter plot for distribution bridge features
    """
    filename = "results/" + name + "featureBoxplot.png"
    plt.figure(figsize=(10, 10))
    plt.style.use("fivethirtyeight")
    plt.title("Box plot")
    sns.boxplot(x='cluster', y='value', data=dfMelt, color='#99c2a2')
    plt.savefig(filename)

# Analysis of the clusters:
def characterize_clusters(dataScaled,
                         listOfParameters,
                         clusterName='cluster',
                         x=''):
    """
    Description:
        Characterize clusters by distribution of each features

    Args:
       dataScaled (dataframe)
       listOfParameters (list)

    Returns:
    """
    listOfParameters.append(clusterName)
    df = dataScaled[listOfParameters]
    clusters = list(df[clusterName].unique())
    listOfParameters.remove(clusterName)
    placeHolder = list(range(0, len(listOfParameters)))
    for cluster in clusters:
        tempDf = df[df[clusterName] == cluster]
        fig = plt.figure(figsize=(10, 7))
        data = list()
        minimums = list()
        maximums = list()
        means = list()
        medians = list()
        stdDevs = list()
        for parameter in listOfParameters:
            values = np.array(tempDf[parameter])
            data.append(values)
            minVal = min(values)
            maxVal = max(values)
            medianVal = np.median(values)
            meanVal = np.mean(values)
            stdDevVal = np.mean(values)

            minimums.append(minVal)
            maximums.append(maxVal)
            medians.append(medianVal)
            means.append(meanVal)
            stdDevs.append(stdDevVal)

        # box plot
        fig = plt.figure(figsize=(10, 7))

        # Creating axes instance
        #ax = fig.add_axes([0, 0, 1, 1])
        ax = fig.add_subplot(111)
        ax.set_xticklabels(listOfParameters)
        bp = ax.boxplot(data)
        plt.xticks(placeHolder, listOfParameters, rotation=45)
        filename = 'results/' + 'Cluster' + str(cluster)
        plt.savefig(filename, bbox_inches='tight')

        # Cluster dataframe
        dataFrame = pd.DataFrame({'mean': means,
                                  'medians': medians,
                                  'maximums': maximums,
                                  'minimums': minimums,
                                  'stdDev': stdDevs})

        print("Cluster: ", cluster)
        print("{:<4} {:<6} {:<7} {:<8} {:<8}".format('Mean',
                                                     'Median',
                                                     'Maximums',
                                                     'Minimums',
                                                     'StdDev'))
        print("="*37)

        for a, b, c, d, e in zip(means,
                                 medians,
                                 maximums,
                                 minimums,
                                 stdDevs):

            print ("{:.2f} {:.2f}   {:.2f}     {:.2f}     {:.2f}".format(a,
                                                                         b,
                                                                         c,
                                                                         d,
                                                                         e))
        print("\n")

# Confusion matrix
def print_confusion_matrix(cm):
    """
    Description:
        Plots confusion matrix on validation set
    """
    indexList = list()
    columnList = list()

    for row in range(0, np.shape(cm)[0]):
        indexString = 'Actual' + str(row+1)
        columnString = 'Predicted' + str(row+1)
        indexList.append(indexString)
        columnList.append(columnString)

    dfCm = pd.DataFrame(cm,
                        index=indexList,
                        columns=columnList)

    plt.subplots(figsize=(8, 8))
    sns.heatmap(dfCm, annot=True, fmt='g', cmap='BuPu')
    plt.savefig("ConfusionMatrix.png")

# Logistic Regression
def log_regression(df, X, y):
    """
    Description:
        Performs training testing split
        Perform multi-nomial logistic regression

    Args:
        df (dataframe)

    Returns:
        model (sklearn model)
    """
    trainX, testX, trainy, testy = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

    model = LogisticRegression(multi_class="multinomial",
                               random_state=0)

    model.fit(trainX, trainy)
    prediction = model.predict(testX)
    cm = confusion_matrix(testy, prediction)
    cr = classification_report(testy, prediction)
    r2 = r2_score(testy, prediction)
    print("\n Intercept: \n", model.intercept_)
    print("\n Coef: \n", model.coef_)
    print("\n Classes: ", model.classes_)
    print("\n Classification: \n", cr)
    print("\n R Square: ", r2)
    print("\n Confusion matrix: \n", cm)
    print_confusion_matrix(cm)
    return model

def evaluate_ANOVA(dataScaled, columns, lowestCount):
    """
    Description:
        Compute ANOVA table for all features in all
        Clusters. And, returns ANOVA to find essential features
        in the cluster analysis

    Args:
        dataScaled (dataframe)
        columns (list)
        lowestCount (integer)

    Returns:
        anovaDf (dataframe)
    """
    pvalues = list()
    features = list()
    sumsqs = list()
    dfs = list()
    fvalues = list()
    tukeys = list()

    for col in columns:
        temp = defaultdict()
        for rows in dataScaled.groupby('cluster')[col]:
            cluster, records = rows
            temp[cluster] = np.random.choice(list(records),
                                             lowestCount)
        tempDf = pd.DataFrame.from_dict(temp)
        anovaTable, tukey = ANOVA(tempDf, col)

        pvalue = anovaTable['PR(>F)'][0]
        pvalues.append(pvalue)

        sumsq = anovaTable['sum_sq'][0]
        sumsqs.append(sumsq)

        df = anovaTable['df'][0]
        dfs.append(df)

        F = anovaTable['F'][0]
        fvalues.append(F)

        features.append(col)

        tukeys.append(tukey)

    anovaDf = pd.DataFrame(columns=['Attribute',
                                    'sum_sq',
                                    'df',
                                    'F',
                                    'p-value'])
    anovaDf['Attribute'] = features
    anovaDf['sum_sq'] = sumsqs
    anovaDf['df'] = dfs
    anovaDf['F'] = fvalues
    anovaDf['p-value'] = pvalues
    return anovaDf, tukeys

def is_zero(value):
    if value == 0:
        return True
    else:
        return False

def is_negative(value):
    if value < 0:
        return True
    else:
        return False
def is_one(value):
    if value == 1:
        return True
    else:
        return False

def is_low(value):
    if value > 0 and value < 0.125:
        return True
    else:
        return False

def is_medium(value):
    if value > 0.125 and value < 0.33:
        return True
    else:
        return False

def is_high(value):
    if value >= 0.33:
        return True
    else:
        return False

#TODO:
    # Compare:
        # Histogram for all the components 
        # (intervention)

def provide_label(sub, deck, sup):
    """
    Description:
        Return the label for the value of
    subsructure, superstructure, and deck
    """
    componentDict = {0:'Substructure',
                     1:'Deck',
                     2:'Superstructure'}

    values = [sub, deck, sup]
    labels = list()
    for num in range(len(values)):
        value = float(values[num])
        if is_zero(value):
            label = 'No ' + componentDict[num]
        elif is_low(value):
            #label = 'Low ' + componentDict[num]
            label = 'Yes ' + componentDict[num]
        elif is_negative(value):
            #label = 'No ' + componentDict[num]
            label = 'Yes ' + componentDict[num]
        elif is_medium(value):
            #label = 'Medium ' + componentDict[num]
            label = 'No ' + componentDict[num]
        elif is_high(value):
            #label = 'High ' + componentDict[num]
            label = 'Yes ' + componentDict[num]
        else:
            label = 'Error ' + componentDict[num]
        labels.append(label)
    return labels

# TODO:
    # Fix this function to create a binary cluster

#def semantic_labeling_utility(record):
#    """
#    Description:
#       Utility to assign a label depending on the values
#    """
#    sub, deck, sup = record
#    if sub == 0 and deck == 0 and sup == 0:
#        label = "No intervention"
#    elif sub == 1 and deck == 1 and sup == 1:
#        label = 'All intervention'
#    elif sub > 0 and deck > 0 and sup > 0:
#        label = provide_label(sub, deck, sup)
#    else:
#        label = "Other intervention"
#        label = provide_label(sub, deck, sup)
#    return label

def semantic_labeling_utility(record):
    """
    Description:
       Utility to assign a label depending on the values
    """
    sub, deck, sup = record
    if sub == 0 and deck == 0 and sup == 0:
        label = "No intervention"
    elif sub == 1 and deck == 1 and sup == 1:
        label = 'All intervention'
    elif sub > 0 and deck > 0 and sup > 0:
        label = provide_label(sub, deck, sup)
    else:
        label = "Other intervention"
        label = provide_label(sub, deck, sup)
    return label

def semantic_labeling(features, name=""):
    """
    Description:
       Assign a semantic label
    """
    labels = list()
    # printng structure numbers for High substructure
    #print("\nPrinting structure numbers for high substructure")
    for index, record in features.iterrows():
        subInt = record['subNumberIntervention']
        deckInt = record['deckNumberIntervention']
        supInt = record['supNumberIntervention']

        label = semantic_labeling_utility([subInt,
                                          deckInt,
                                          supInt])
        if type(label) is type(list()):
            label = ' - '.join(label)
            # TODO: 
                # Print out structure numbers with High substructure
            #if label == 'High Substructure - No Deck - No Superstructure':
                #print("\n")
                #print(label)
                #print(record)
        labels.append(label)
    return labels

def three_d_scatterplot(dataScaled, name=''):
    """
    Description:
        Creates a 3d scatter plot of the attributes
        provided
    Args:
        dataScaled (dataframe)
    Returns:
        saves a three 3d scatter plot with .html extention
    """
    filename = "results/" + name + "3d.html"
    title = "3D representation of clusters for the state " + name
    fig = px.scatter_3d(dataScaled,
                        x='subNumberIntervention',
                        y='supNumberIntervention',
                        z='deckNumberIntervention',
                        color='cluster',
                        title=title)

    plotly.offline.plot(fig, filename=filename)

def kmeans_clustering(dataScaled, listOfParameters, kmeans_kwargs, state):
    """
    Description:
        Performs selection of optimal clusters and kmeans
        and returns an updated dataframe with cluster membership
        for each bridge

    Args:
       dataScaled (dataframe)
       listOfParameters (dataframe)
       kmeans_kwargs (dictionary)

    Returns:
        dataScaled (dataframe)
    """
    sse = list()
    for k in tqdm(range(1, 11), desc='Computing Elbow'):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(dataScaled[listOfParameters])
        sse.append(kmeans.inertia_)

    # Plot SSE
    plot_elbow(sse)

    # Find optimum number of clusters
    kl = KneeLocator(range(1, 11),
                    sse,
                    curve='convex',
                    direction='decreasing')

    # Print elbow and silhouette coefficient
    print("\n Optimal number of cluster (Elbow Coefficient): ", kl.elbow)

    # Use K-means with optimal numbe of cluster
    finalKmeans = KMeans(n_clusters=kl.elbow, **kmeans_kwargs)
    finalKmeans.fit(dataScaled[listOfParameters])
    centroids = finalKmeans.cluster_centers_
    labels = finalKmeans.labels_

    print("\n Centroids: ", centroids)
    counts = Counter(labels)
    lowestCount = min(counts.values())

    print("\n Number of members in cluster:", counts)
    slabels = semantic_labeling(centroids, name='')
    print("\n Printing semantic labeling:")
    newNames = dict(zip(list(range(len(counts.keys()))), slabels))

    newLabels = [newNames[label] for label in labels]

    # Save cluster as columns
    dataScaled['cluster'] = list(newLabels)

    # Create 3D-clustering of the data
    three_d_scatterplot(dataScaled, name=state)
    return dataScaled, lowestCount, centroids, counts
