"""------------------------------------------------->
Description: Maintenance Model
    This maintenance model only takes into account
    bridges all bridges in midwest.
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

#from decisionmethod import decision_tree
from decision_tree import *
from kmeans import *
from gplot import *

def scale(values):
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
        Generate codify
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

                        "deckDeteriorationScore",
                        "subDeteriorationScore",
                        "supDeteriorationScore"
                        ]


    # Select final columns:
    columnsFinal = [
                    #"deck",
                    #"substructure",
                    #"superstructure",
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
                    #"deckDeteriorationScore",
                    #"subDeteriorationScore",
                    #"supDeteriorationScore"
                ]


    dataScaled = normalize(df, columnsNormalize)
    dataScaled = dataScaled[columnsFinal]
    dataScaled = remove_null_values(dataScaled)

    # Data Scaled
    features = ["supNumberIntervention",
                "subNumberIntervention",
                "deckNumberIntervention"]

    sLabels = semantic_labeling(dataScaled[features], name="")
    dataScaled['cluster'] = sLabels

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

    labels = ['No Substructure - High Deck - No Superstructure',
              'High Substructure - No Deck - No Superstructure',
              'No Substructure - No Deck - High Superstructure']

    kappaValues = list()
    accValues = list()
    featImps = list()
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

        # State column:
        dataScaled['state'] = [state]*len(dataScaled)

        # Modeling features and groundtruth:
        X, y = dataScaled[columnsFinal], dataScaled['label']

        # Summarize distribution before:
        print("\n Distribution of the clusters before oversampling: ", Counter(y))

        # Oversampling:
        oversample = SMOTE()
        print("\n Oversampling (SMOTE) ...")
        X, y = oversample.fit_resample(X, y)

        # Summarize distribution:
        print("\n Distribution of the clusters after oversampling: ", Counter(y))

        # Return to home directory:
        kappaValue, accValue, featImp = decision_tree(X, y)
        kappaValues.append(kappaValue)
        accValues.append(accValue)
        featImps.append(featImp)

    sys.stdout.close()
    os.chdir(currentDir)

    return dataScaled, labels, kappaValues, accValues, featImps

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

    modelName = 'mt_midwest_wo_det_cr'
    #csvfiles = ['nebraska']
    listOfKappaValues = list()
    listOfAccValues = list()
    listOfLabels = list()
    listOfStates = list()
    listOfCounts = list()
    listOfDataFrames = list()
    listOfFeatureImps = list()

    for filename in csvfiles:
         filename = filename+'_deep'
         dataScaled, sLabel, kappaValues, accValues, featImps = maintenance_pipeline(filename)
         listOfLabels.append(sLabel)
         listOfStates.append([filename[:-5]]*3)
         listOfDataFrames.append(dataScaled)
         listOfKappaValues.append(kappaValues)
         listOfAccValues.append(accValues)
         listOfFeatureImps.append(featImps)

    summaryfilename = modelName + '.txt'
    sys.stdout = open(summaryfilename, "w")

    # Refactor
    oneListOfFeaturesImp = list()
    for forStates in listOfFeatureImps:
        maps = list()
        for tempMap in forStates:
            maps.append(tempMap[0])
        oneListOfFeaturesImp.append(maps)

    ## change the orientation:
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

    # TODO: Refactor
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
    accTitle='Acc values with respect to states'
    kappa='kappa'
    acc='accuracy'
    state='state'
    plot_barchart(metricsdf, kappa, state, kappaTitle)
    plot_barchart(metricsdf, acc, state, accTitle)
    sys.stdout.close()

if __name__=='__main__':
    main()