"""------------------------------------------------->
Description: Maintenance Model
    This maintenance model only takes into account
    bridges all bridges in midwest.

Notes:
    This whole idea of the decision tree??

Author: Akshay Kale
Date: August 9th, 2021
<------------------------------------------------"""

# Data structures
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict
from collections import OrderedDict

# System Libraries
import os
import sys

# ML
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

# Custom files
from decision_tree import *
from kmeans import *
from gplot import *

def scale(values):
    """
    Description:
        A function to scale the values
    Args:
        values
    Return:
        Scaled values
    """
    newValues = list()
    minVal = min(values)
    maxVal = max(values)
    rangeMin = 1
    rangeMax = 5
    for val in values:
        valNormalized = ((rangeMax - rangeMin) * (val- minVal) / (maxVal - minVal)) + rangeMin
        newValues.append(valNormalized)
    return newValues

def codify(listOfValues, dictionary):
    """
    Description:
         Codify values according to the provides
         dictionary
    """
    newListOfValues = list()
    for val in listOfValues:
        newListOfValues.append(dictionary.get(val))
    return newListOfValues

def generate_dictionary(uniques):
    """
    Description:
        Generate sankey dictionary
    """
    sankeyDict = defaultdict()
    for index, label in enumerate(uniques):
       sankeyDict[label] = index
    return sankeyDict

# TODO: Number 1?
#def generate_heat_map(listOfStates, listOfClusters, listOfFeatures):
#    """
#    Description:
#        Generate data for heatmap
#
#    Args:
#        states (list of list)
#        clusters (list of list)
#        features (list of dictionary)
#
#    Returns:
#        dataframe (a pandas dataframe)
#    """
#    data = list()
#    heatMapDict = defaultdict(list)
#    for clusters, features, states in zip(listOfClusters,
#                                       listOfFeatures,
#                                       listOfStates):
#        for clus, feat, state in zip(clusters,
#                                     features,
#                                     states):
#            heatMapDict[state].append((clus, feat))
#
#    states = heatMapDict.keys()
#    for state in states:
#        clusterVals = heatMapDict[state]
#        tempData = list()
#        for val in clusterVals:
#            clus, featMap = val
#            tempSeries = pd.Series(data=featMap,
#                                   index=featMap.keys(),
#                                   name=clus)
#            tempData.append(tempSeries)
#        tempDf = pd.concat(tempData, axis=1).reset_index()
#        tempDf.set_index('index', inplace=True)
#        data.append(tempDf)
#    return states, data
#

def generate_heat_map(listOfStates, listOfClusters, listOfFeatures):
    """
    Description:
        Generate data for heatmap

    Args:
        states (list of list)
        clusters (list of list)
        features (list of dictionary)

    Returns:
        dataframe (a pandas dataframe)
    """
    data = list()
    heatMapDict = defaultdict(list)
    for clusters, features, states in zip(listOfClusters,
                                       listOfFeatures,
                                       listOfStates):
        for clus, feat, state in zip(clusters,
                                     features,
                                     states):
            heatMapDict[state].append((clus, feat))

    states = heatMapDict.keys()
    for state in states:
        clusterVals = heatMapDict[state]
        tempData = list()
        for val in clusterVals:
            clus, featMap = val
            tempSeries = pd.Series(data=featMap,
                                   index=featMap.keys(),
                                   name=clus)
            tempData.append(tempSeries)
        #tempDf = pd.concat(tempData, axis=1).reset_index()
        #tempDf.set_index('index', inplace=True)
        data.append(tempData)
    #print(data)
    return states, data

def generate_sankey_data(listOfStates, listOfClusters, listOfFeatures):
    """
    Description:
        Generate data for sankey plot
    Args:
        states (list of list)
        clusters (list of list)
        features (list of dictionary)

    Returns:
        dataframe (a pandas dataframe)
    """
    sources = list()
    targets = list()
    values = list()
    uniques = set()

    for states, features, clusters in zip (listOfStates, listOfFeatures, listOfClusters):
        for state, cluster, feature in zip(states, clusters, features):
            # Create a dictionary of keys and ranks
            feat = OrderedDict()
            for key, value in feature.items():
                feat[key] = value

            for value, key in enumerate(feat.keys()):
                set1 = (state, key)
                set2 = (key, cluster)

                sources.append(set1[0])
                targets.append(set1[1])
                values.append(1)

                sources.append(set2[0])
                targets.append(set2[1])
                values.append(2)
                values.append(value)

                uniques.add(state)
                uniques.add(cluster)
                uniques.add(key)

    return sources, targets, values, uniques

def flatten_list(nestedList):
    """
    Description:
        Function to flatten the list
    Args:
        nestedList (a list of nested list)
    Returns:
        newNestedList ( a revised nested list)
    """
    newNestedList = list()
    for valuesPerState in newNestedList:
        if len(np.shape(valuesPerState)) != 1:
            for value in valuesPerState:
                newNestedList.append(value[0])
        else:
            for values in valuesPerState:
                newNestedList.append(value)
    return newNestedList

def maintenance_pipeline(state):
    """
    Description:
        Pipeline for determining future maintenance of the bridges
    """

    # Creating directory
    csvfilename = state + '.csv'
    directory = state + 'OutputsTICR'

    # Create a state folder/ Change directory and then come out
    df = pd.read_csv(csvfilename, index_col=None, low_memory=False)

    # Change directory
    os.mkdir(directory)
    currentDir = os.getcwd()

    # Create results folders
    newDir = currentDir + '/' + directory
    os.chdir(newDir)
    modelOutput = state + 'ModelSummary.txt'

    sys.stdout = open(modelOutput, "w")
    print("\n state: ", state)
    resultsFolder = 'results'
    modelsFolder = 'models'
    os.mkdir(resultsFolder)
    os.mkdir(modelsFolder)

    # Remove null values:
    df = df.dropna(subset=['deck',
                           'substructure',
                           'superstructure',
                           'deckNumberIntervention',
                           'subNumberIntervention',
                           'supNumberIntervention',
                           ])

    # Only take the last record to avoid double counting:
    df = remove_duplicates(df)

    # Remove values encoded as N:
    df = df[~df['deck'].isin(['N'])]
    df = df[~df['substructure'].isin(['N'])]
    df = df[~df['superstructure'].isin(['N'])]
    df = df[~df['material'].isin(['N'])]
    df = df[~df['scourCriticalBridges'].isin(['N', 'U', np.nan])]
    df = df[~df['deckStructureType'].isin(['N', 'U'])]

    # Fill the null values with -1:
    df.snowfall.fillna(value=-1, inplace=True)
    df.precipitation.fillna(value=-1, inplace=True)
    df.freezethaw.fillna(value=-1, inplace=True)
    df.toll.fillna(value=-1, inplace=True)
    df.designatedInspectionFrequency.fillna(value=-1, inplace=True)
    df.deckStructureType.fillna(value=-1, inplace=True)
    df.typeOfDesign.fillna(value=-1, inplace=True)

    # Normalize features:
    columnsNormalize = [
                        "deck",
                        "yearBuilt",
                        "superstructure",
                        "substructure",
                        "averageDailyTraffic",
                        "avgDailyTruckTraffic",
                        "supNumberIntervention",
                        "subNumberIntervention",
                        "deckNumberIntervention",
    # New
                        "latitude",
                        "longitude",
                        "skew",
                        "numberOfSpansInMainUnit",
                        "lengthOfMaximumSpan",
                        "structureLength",
                        "bridgeRoadwayWithCurbToCurb",
                        "operatingRating",
                        "scourCriticalBridges",
                        "lanesOnStructure",

                        "freezethaw",
                        "snowfall",
                        "designatedInspectionFrequency",

                        "deckDeteriorationScore",
                        "subDeteriorationScore",
                        "supDeteriorationScore"
                        ]
# 

    # Select final columns:
    columnsFinal = [
    #               "deck",
    #               "substructure",
    #               "superstructure",
                    "structureNumber",
                    "yearBuilt",
                    "averageDailyTraffic",
                    "avgDailyTruckTraffic",
                     "material",
                     "designLoad",
                    "snowfall",
                    "freezethaw",
                    "supNumberIntervention",
                    "subNumberIntervention",
                    "deckNumberIntervention",
                    "latitude",
                    "longitude",
                    "skew",
                    "numberOfSpansInMainUnit",
                    "lengthOfMaximumSpan",
                    "structureLength",
                    "bridgeRoadwayWithCurbToCurb",
                    "operatingRating",
                    "scourCriticalBridges",
                    "lanesOnStructure",
                     "toll",
                     "designatedInspectionFrequency",
                     "deckStructureType",
                     "typeOfDesign",
    #               "deckDeteriorationScore",
    #               "subDeteriorationScore",
    #               "supDeteriorationScore"
                ]

    #dataScaled = normalize(df, columnsNormalize)
    dataScaled = df[columnsFinal]
    #dataScaled = dataScaled[columnsFinal]
    #print(dataScaled.columns)

    # TODO: Do one hot encoding here
    columnsHotEncoded = ['material',
                         "toll",
                         "designLoad",
                         "deckStructureType",
                         "typeOfDesign"]

    dataScaled = oneHot(dataScaled, columnsHotEncoded)
    dataScaled = remove_null_values(dataScaled)
    dataScaled = convert_geo_coordinates(dataScaled,
                                         ['longitude',
                                         'latitude'])

    columnsFinal = list(dataScaled.columns)
    columnsFinal.remove('material')
    columnsFinal.remove('toll')
    columnsFinal.remove('designLoad')
    columnsFinal.remove('deckStructureType')
    columnsFinal.remove('typeOfDesign')

    # Apply recursive feature elimination
    # Data Scaled
    features = ["structureNumber",
                "supNumberIntervention",
                "subNumberIntervention",
                "deckNumberIntervention"]

    print("\nPrinting the labels")
    #TODO: Clean data up until here:
    # print(dataScaled.head())
    sLabels = semantic_labeling(dataScaled[features],
                                name="")
    columnsFinal.remove('structureNumber')
    features.remove('structureNumber')
    dataScaled = dataScaled[columnsFinal]

    dataScaled['cluster'] = sLabels
    newFeatures = features + ['cluster']
    plot_scatterplot(dataScaled[newFeatures], name="cluster")

    print("\n")
    print(dataScaled['cluster'].unique())
    print("\n")

    # Analysis of Variance:
    #anovaTable, tukeys =  evaluate_ANOVA(dataScaled, features, lowestCount)
    #print("\nANOVA: \n", anovaTable)
    #print("\nTukey's : \n")
    #for result in tukeys:
    #    print(result)
    #    print('\n')

    # Characterizing the clusters:
    characterize_clusters(dataScaled, features)

    # Remove columns:
    columnsFinal.remove('supNumberIntervention')
    columnsFinal.remove('subNumberIntervention')
    columnsFinal.remove('deckNumberIntervention')

    #labels = ['No Substructure - High Deck - No Superstructure',
    #          'High Substructure - No Deck - No Superstructure',
    #          'No Substructure - No Deck - High Superstructure']

    #TODO: Why do I have these label writen all intervention?
    labels = ['No Substructure - YesDeck - No Superstructure',
              'YesSubstructure - No Deck - No Superstructure',
              'No Substructure - No Deck - YesSuperstructure']

    #labels = ['All intervention',
    #          'All intervention',
    #          'All intervention']

    kappaValues = list()
    accValues = list()
    featImps = list()
    decisionModels = list()

    ## TODO: This loop takes into account defined labels
    for label in labels:
        print("\nCategory (Positive Class): ", label)
        print("----------"*5)
        dataScaled = create_labels(dataScaled, label)
        clusters = Counter(dataScaled['label'])
        listOfClusters = list()
        for cluster in clusters.keys():
            numOfMembers = clusters[cluster]
            if numOfMembers < 15:
                listOfClusters.append(cluster)
        dataScaled = dataScaled[~dataScaled['label'].isin(listOfClusters)]
        # Divide these the dataset into two categories:
            # Negative and Positive
            # Negative: Show YearBuilt
            # Show YearBuilt for negative and positive classes 

        # State column:
        dataScaled['state'] = [state]*len(dataScaled)

        # Modeling features and groundtruth:
        X, y = dataScaled[columnsFinal], dataScaled['label']

        # Summarize distribution before:
        print("\n Distribution of the clusters before oversampling: ", Counter(y))

        # Oversampling:
        oversample = SMOTE()
        # TODO: Run a independent test / Bayesian test: 
            # What is the probability of getting a year built given the cluster is 0 or 1?
        # print(dataScaled.columns())
        neg = dataScaled[dataScaled['label'] == 'negative']
        pos = dataScaled[dataScaled['label'] == 'positive']

        # Create a dictionary:
        negativeDict = defaultdict()
        positiveDict = defaultdict()

        print("Index and length of the rows")
        for index, row in neg.groupby(['yearBuilt']):
            negativeDict[index] = len(row)

        for index, row in pos.groupby(['yearBuilt']):
            positiveDict[index] = len(row)

        #TODO:
        #print(positiveDict)
        #print(negativeDict)

        #plot_barchart1(positiveDict, 'barchart positive')
        #plot_barchart1(negativeDict, 'barchart negative')

        #print(neg.groupby(['yearBuilt']).count())
        #print(pos.groupby(['yearBuilt']).count())
        #print(pos['yearBuilt'].head())
        #plot_barchart(dataScaled,
        #              'yearBuilt',
        #              'label',
        #              'barchart1')

        #print("\n Oversampling (SMOTE) ...")
        X, y = oversample.fit_resample(X, y)

        # Summarize distribution:
        print("\n Distribution of the clusters after oversampling: ", Counter(y))

        # Return to home directory:
            # Overhere, if the model is passed here then we can print the trees
            # in the following main function
        kappaValue, accValue, featImp, models = decision_tree(X,
                                                              y,
                                                              columnsFinal,
                                                              label)
        kappaValues.append(kappaValue)
        accValues.append(accValue)
        featImps.append(featImp)
        decisionModels.append(models)


        # TODO: working on the leaves
        # models.append(leaves) # models ->> change into leaves 
        # list of model is being is used to save models from decision tree
        # Find leaves first

    #print(dataScaled.head())
    sys.stdout.close()
    os.chdir(currentDir)

    return dataScaled, labels, kappaValues, accValues, featImps, decisionModels

# Driver function
def main():
    # States
    csvfiles = [
                "nebraska",
                "kansas",
                "indiana",
                "illinois",
                "ohio",
                "wisconsin",
                "missouri",
                "minnesota"
                ]

    modelName = 'testing'
    csvfiles = ['nebraska']

    listOfKappaValues = list()
    listOfAccValues = list()
    listOfLabels = list()
    listOfStates = list()
    listOfCounts = list()
    listOfDataFrames = list()
    listOfFeatureImps = list()
    listOfModels = list()

    for filename in csvfiles:
         filename = filename+'_deep'
         dataScaled, sLabel, kappaValues, accValues, featImps, decisionModels = maintenance_pipeline(filename)
         listOfLabels.append(sLabel)
         listOfStates.append([filename[:-5]]*3)
         listOfDataFrames.append(dataScaled)
         listOfKappaValues.append(kappaValues)
         listOfAccValues.append(accValues)
         listOfFeatureImps.append(featImps)

    summaryfilename = modelName + '.txt'
    sys.stdout = open(summaryfilename, "w")

    #TODO: Refactor
        # Simplfy the some of the hard coded segments of the program
    oneListOfFeaturesImp = list()
    for forStates in listOfFeatureImps:
        maps = list()
        for tempMap in forStates:
            maps.append(tempMap[0])
        oneListOfFeaturesImp.append(maps)

    # Change the orientation:
    states = list()
    clusternames = list()
    countstemp = list()

    ## print the values:
    for  slabel, state, counts in zip(listOfLabels,
                                      listOfStates,
                                      listOfCounts):
        counts = dict(counts).values()
        for label, item1, count in zip(slabel, state, counts):
            states.append(state)
            clusternames.append(label)
            countstemp.append(count)
    to_csv(listOfDataFrames)

    # printing acc, kappa, and labels
    newlistofkappavalues = list()
    newlistofaccvalues = list()
    newlistoflabels = list()
    newlistofstates = list()
    newlistoffeatimps = list()

    # TODO: Refactor the series of for loops
    for valuesperstate in listOfKappaValues:
        for values in valuesperstate:
            entropy, gini = values
            newlistofkappavalues.append(entropy)

    for valuesperstate in listOfAccValues:
        for values in valuesperstate:
            entropy, gini = values
            newlistofaccvalues.append(entropy)

    for valueperstate in listOfLabels:
        for value in valueperstate:
            newlistoflabels.append(value)

    for valueperstate in listOfStates:
        for value in valueperstate:
            newlistofstates.append(value)

    for valuesperstate in oneListOfFeaturesImp:
        for value in valuesperstate:
            newlistoffeatimps.append(value)

    # Create a new dataframe 
    metricsdf = pd.DataFrame({'state': newlistofstates,
                              'kappa': newlistofkappavalues,
                              'accuracy': newlistofaccvalues,
                              'cluster': newlistoflabels})

    # Plot heatmap
    col, data = generate_heat_map(listOfStates, listOfLabels, oneListOfFeaturesImp)
    for col, val in zip(col, data):
        fname = col + '_heatmap.csv'
        #val.to_csv(fname)
        #print('\nCluster:', col)
        # TODO: create barchart here
        print("\n printing heatmap")
        plot_barchart_sideway(val, col)
        # Printing in testing.txt
        print(val)
        plot_heatmap(val, col)

    # Plot sankey
    sources, targets, values, labels = generate_sankey_data(listOfStates, listOfLabels, oneListOfFeaturesImp)
    sankeyDict = generate_dictionary(labels)
    sources = codify(sources, sankeyDict)
    targets = codify(targets, sankeyDict)

    values = scale(values)
    labels = list(labels)
    title = 'Important features with respect to states and cluster'
    plot_sankey_new(sources, targets, values, labels, title)

    # Plot barchart
    kappaTitle='Kappa values with respect to states'
    accTitle='Accuracy values with respect to states'
    kappa='kappa'
    acc='accuracy'
    state='state'
    plot_barchart(metricsdf, kappa, state, kappaTitle)
    plot_barchart(metricsdf, acc, state, accTitle)
    sys.stdout.close()

if __name__=='__main__':
    main()
