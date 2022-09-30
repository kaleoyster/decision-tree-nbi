"""------------------------------------------------->
Description: Maintenance Model
    This maintenance model only takes into account
    bridges all bridges in midwest.

Author: Akshay Kale
Date: August 9th, 2021
<------------------------------------------------"""

# Data structures
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
import os
import sys
import pandas as pd
import numpy as np

# ML
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTEN
#from sklearn import preprocessing

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
        print("\n Testing features")
        print(features)

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
    sources = []
    targets = []
    values = []
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
    #print("Scaled features", dataScaled.columns)
    dataScaled = df[columnsFinal]
    dataScaled = dataScaled[columnsFinal]


    # TODO: Do one hot encoding here,
    tempColumnsHotEncoded = {'material': 'CatMaterial',
                         "toll": 'CatToll',
                         "designLoad": 'CatDesignLoad' ,
                         "deckStructureType": 'CatDeckStructureType',
                         "typeOfDesign":'CatTypeOfDesign'}

    dataScaled.rename(columns=tempColumnsHotEncoded, inplace=True)
    columnsHotEncoded = tempColumnsHotEncoded.values()
    dataScaled = one_hot(dataScaled, columnsHotEncoded)
    #TODO: Something went wrong over here, when you apply normalize
    dataScaled = remove_null_values(dataScaled)

    dataScaled = convert_geo_coordinates(dataScaled,
                                         ['longitude',
                                         'latitude'])

    columnsFinal = list(dataScaled.columns)
    columnsFinal.remove('CatMaterial')
    columnsFinal.remove('CatToll')
    columnsFinal.remove('CatDesignLoad')
    columnsFinal.remove('CatDeckStructureType')
    columnsFinal.remove('CatTypeOfDesign')

    #TODO: Apply recursive feature elimination here.
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
    structureNumber = dataScaled['structureNumber']
    # We cannot take the X and y values after the oversampling
    all_data = dataScaled
    structure_number = dataScaled['structureNumber']
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

    #TODO: Why do I have these label with all interventions?
    labels = ['No Substructure - Yes Deck - No Superstructure',
              'Yes Substructure - No Deck - No Superstructure',
              'No Substructure - No Deck - Yes Superstructure']

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

        # State column:
        dataScaled['state'] = [state]*len(dataScaled)

        # Modeling features and groundtruth:
        X, y = dataScaled[columnsFinal], dataScaled['label']

        # Summarize distribution before:
        print("\n Distribution of the clusters before oversampling: ", Counter(y))
        all_data = dataScaled
        all_data['structureNumber'] = structureNumber

        # Oversampling techniques:
        oversample = SMOTE()
        oversample = SMOTEN(random_state=0)
        #oversample = SMOTNC()

        # Undersampling:
        #undersample = RandomUnderSampler(sampling_strategy='auto')

        # SMOTENC

        # TODO: Run a independent test / Bayesian test: 
            # What is the probability of getting a year built given the cluster is 0 or 1?
        # print(dataScaled.columns())
        neg = dataScaled[dataScaled['label'] == 'negative']
        pos = dataScaled[dataScaled['label'] == 'positive']
        #all_data =  dataScaled

        # Create a dictionary:
        negativeDict = defaultdict()
        positiveDict = defaultdict()

        print("Index and length of the rows")
        for index, row in neg.groupby(['yearBuilt']):
            negativeDict[index] = len(row)

        for index, row in pos.groupby(['yearBuilt']):
            positiveDict[index] = len(row)

        #TODO:
        #plot_barchart1(positiveDict, 'barchart positive')
        #plot_barchart1(negativeDict, 'barchart negative')

        #print(neg.groupby(['yearBuilt']).count())
        #print(pos.groupby(['yearBuilt']).count())
        #print(pos['yearBuilt'].head())
        #plot_barchart(dataScaled,
        #              'yearBuilt',
        #              'label',
        #              'barchart1')

        print("\n Oversampling (SMOTE) ...")
        #temp_X, temp_y = undersample(X, y)
        X, y = oversample.fit_resample(X, y)
        #X, y = undersample.fit_resample(X, y)

        # TODO: Undersampling

        # Summarize distribution:
        print("\n Distribution of the clusters after oversampling: ", Counter(y))

        # Return to home directory:
            # Overhere, if the model is passed here then we can print the trees
            # in the following main function
        print("\n Test, length of X and y : ", len(X), len(y))
        kappaValue, accValue, featImp, models = decision_tree(X,
                                                              y,
                                                              columnsFinal,
                                                              label,
                                                              all_data)
        print("\n List of final columns", columnsFinal)
        kappaValues.append(kappaValue)
        accValues.append(accValue)
        featImps.append(featImp)
        decisionModels.append(models)

    sys.stdout.close()
    os.chdir(currentDir)

    return dataScaled, labels, kappaValues, accValues, featImps, decisionModels

# Driver function
def main():
    """
    driver method
    """
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

    modelName = 'modelNebraska'
    csvfiles = ['nebraska']

    listOfKappaValues = []
    listOfAccValues = []
    listOfLabels = []
    listOfStates = []
    listOfCounts = []
    listOfDataFrames = []
    listOfFeatureImps =[]
    listOfModels = []

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
    oneListOfFeaturesImp = []
    for forStates in listOfFeatureImps:
        maps = []
        for tempMap in forStates:
            maps.append(tempMap[0])
        oneListOfFeaturesImp.append(maps)

    # Change the orientation:
    states = []
    clusternames = []
    countstemp = []

    ## print the values:
    for  slabel, state, counts in zip(listOfLabels,
                                      listOfStates,
                                      listOfCounts):
        counts = dict(counts).values()
        for label, item1, count in zip(slabel, state, counts):
            states.append(state)
            clusternames.append(label)
            countstemp.append(count)
    #to_csv(listOfDataFrames)

    # printing acc, kappa, and labels
    newlistofkappavalues = []
    newlistofaccvalues = []
    newlistoflabels = []
    newlistofstates = []
    newlistoffeatimps = []

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

    # Plot heatmap and generating csv files
    col, data = generate_heat_map(listOfStates, listOfLabels, oneListOfFeaturesImp)
    heatmapMap = defaultdict(list)
    for col, val in zip(col, data):
        fname = col + '_heatmap.csv'
        tempMap = defaultdict()
        for values in val:
            indx = values.index
            dfName = values.name
            dfVal = list(values)
            heatmapMap['category'].append(dfName)
            for feat, valN in zip(indx, dfVal):
                heatmapMap[feat].append(valN)
        heatmapDf = pd.DataFrame(heatmapMap)
        heatmapDf = heatmapDf.T
        heatmapDf.to_csv(fname)

        # Creating a horizontal barchart
        plot_barchart_sideway(val, col)
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
