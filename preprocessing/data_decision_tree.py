"""
description: A custom script to create a dataset
for decision tree model for:
    1. deterioration
    2. maintenance

author: Akshay Kale
data: May 27th, 2020

TODO:
        1. Create pipeline for deterioration and maintenance
        3. Import precipitation and snowfall data into mongodb
        4. Investigate snowfall and freezethaw data for Nebraska
        5. Create a gaint for loop
            # 1. Compute deterioration
            # 2. Intervention
            # 3. Scores


NOTES:
"""
__author__ = 'Akshay Kale'
__copyright__ = "GPL"
__email__ = "akale@unomaha.edu"

from nbi_data_chef import *

# Driver function
def main():
    nbiDB = get_db()
    collection = nbiDB['nbi']

    # select features:
    fields = {
                "_id":0,
                "year":1,
                "structureNumber":1,
                "yearBuilt":1,
                "yearReconstructured":1,
                "averageDailyTraffic":1,
                "avgDailyTruckTraffic":1,
                "deck":1,
                "substructure":1,
                "superstructure":1,
                "owner":1,
                "maintainanceResponsibility":1,
                "designLoad":1,
                "operatingRating":1,
                "structureLength":1,
                "numberOfSpansInMainUnit":1,
                "scourCriticalBridges":1,
                "material":"$structureTypeMain.kindOfMaterialDesign",
                "wearingSurface":"$structureTypeMain.kindOfDesignConstruction",
            }

    # select states:
    states = ['55'] # Wisconsin

    # years:
    years = [year for year in range(1992, 2017)]

    # process precipitation data
    structBdsMap, structPrecipMap = process_precipitation()

    # process snowfall and freezethaw data
    structSnowMap, structFreezeMap = process_snowfall()

    # query
    individualRecords = query(fields, states, years, collection)
    #individualRecords = sample_records()

    # group records
    groupedRecords = group_records(individualRecords, fields)

    # integrate baseline difference score, precipitation, freezethaw, and snowfall
    individualRecords = integrate_ext_dataset_list(structBdsMap, individualRecords, 'baseDifferenceScore')
    individualRecords = integrate_ext_dataset_list(structPrecipMap, individualRecords, 'precipitation')
    individualRecords = integrate_ext_dataset_list(structSnowMap, individualRecords, 'snowfall')
    individualRecords = integrate_ext_dataset_list(structFreezeMap, individualRecords, 'freezethaw')

    # divide grouped records (works only for grouped records)
    groupedRecords = divide_grouped_records(groupedRecords, fields, 2010, 2017)

    # remove records from specific years (works only for individual records)
    individualRecords = remove_records(individualRecords, 2010, 2017)

    # compute intervention from (year: from and to) 
    groupedRecords = compute_intervention(groupedRecords, from_to_matrix)
    groupedRecords = compute_intervention(groupedRecords, from_to_matrix, component='substructure')
    groupedRecords = compute_intervention(groupedRecords, from_to_matrix, component='superstructure')
    #print("\n printing grouped records: ", groupedRecords)

    # compute deterioration
    groupedRecords = compute_deterioration_slope(groupedRecords, component='deck')
    groupedRecords = compute_deterioration_slope(groupedRecords, component='substructure')
    groupedRecords = compute_deterioration_slope(groupedRecords, component='superstructure')

    # compute deterioration
    # Incomplete function: 
    deckStructDetMap = create_deterioration_dict(groupedRecords,
                                             2010,
                                            'deckDeteriorationScore')

    subStructDetMap = create_deterioration_dict(groupedRecords,
                                             2010,
                                            'substructureDeteriorationScore')

    superStructDetMap = create_deterioration_dict(groupedRecords,
                                             2010,
                                            'superstructureDeteriorationScore')

    # Number of intervention
    deckStructIntMap = create_deterioration_dict(groupedRecords,
                                             2010,
                                            'deckNumberOfInterventions')

    subStructIntMap = create_deterioration_dict(groupedRecords,
                                             2010,
                                            'substructureNumberOfInterventions')

    supStructIntMap = create_deterioration_dict(groupedRecords,
                                             2010,
                                            'superstructureNumberOfInterventions')

    # Scores
    individualRecords = integrate_ext_dataset_list(deckStructDetMap,
                                                   individualRecords,
                                                   'deckDeteriorationScore')

    individualRecords = integrate_ext_dataset_list(subStructDetMap,
                                                   individualRecords,
                                                   'subDeteriorationScore')

    individualRecords = integrate_ext_dataset_list(superStructDetMap,
                                                   individualRecords,
                                                   'supDeteriorationScore')

    # Number of Intervention
    individualRecords = integrate_ext_dataset_list(deckStructIntMap,
                                                   individualRecords,
                                                   'deckNumberIntervention')

    individualRecords = integrate_ext_dataset_list(subStructIntMap,
                                                   individualRecords,
                                                   'subNumberIntervention')

    individualRecords = integrate_ext_dataset_list(supStructIntMap,
                                                   individualRecords,
                                                   'supNumberIntervention')

    # save to the file
    csvFile = 'wisconsin.csv'
    tocsv_list(individualRecords, csvFile)

if __name__=='__main__':
    main()
