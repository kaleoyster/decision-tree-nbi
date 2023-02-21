"""
Description: A custom script to create a dataset for decision tree model for:
    1. Deterioration
    2. Maintenance

Author: Akshay Kale
Date: May 27th, 2020

Last Updated:
    February 16th, 2022

TODO:
    1. Create for deterioration and maintenance?
    2. Monotonously decreasing condition ratings must have negative deterioration score
    3. Import precipitation and snowfall data into mongodb
    4. Investigate snowfall and freezethaw data for Nebraska
    5. Create function to output multiple csv files

NOTES:
"""
__author__ = 'Akshay Kale'
__copyright__ = "GPL"
__email__ = "akale@unomaha.edu"

from nbi_data_chef import *

def categorize_by_adt(adt):
    """
    """
    if adt < 50:
        category = 'Ultra Light'
    elif 51 < adt <= 100:
        category = "Very Light"
    elif 101 < adt <= 1000:
        category = "Light"
    elif 1001 < adt <= 5000:
        category = "Moderate"
    elif adt > 5000:
        category = "High"
    else:
        category = None
    return category


def compute_deck_age(records):
    """
    description: returns the computed deck age
    by computing the difference between the year
    reported and the greater between year reconstructed
    or year built.
    args: records (list of dictionaries)
    return: newRecords (list of new records)
    """
    newRecords = []
    for record in records:
        tempDictionary = {}
        for key, value in zip(record.keys(), record.values()):
            tempDictionary[key] = value
            if key == "yearReconstructed":
                year_built = record['yearBuilt']
                year = record['year']
                if year_built < value:
                    tempDictionary['deckAge'] = year - value
                else:
                    tempDictionary['deckAge'] = year - year_built
        newRecords.append(tempDictionary)
    return newRecords

def compute_age_1(records):
    """
    Description: returns the computed deck age
    by computing the difference between the year
    reported and the greater between year reconstructed
    or year built.

    Args: records (list of dictionaries)

    Return: newRecords (list of new records)
    """
    newRecords = []
    for record in records:
        tempDictionary = {}
        for key, value in zip(record.keys(), record.values()):
            tempDictionary[key] = value
            if key == "yearBuilt":
                year_built = record['yearBuilt']
                year = record['year']
                tempDictionary['age'] = year - year_built
        newRecords.append(tempDictionary)
    return newRecords

def compute_adt_cat(individual_records):
    new_individual_records = []
    for record in individual_records:
        adt = record['averageDailyTraffic']
        cat = categorize_by_adt(adt)
        record['adtCategory'] = cat
        new_individual_records.append(record)
    return new_individual_records

def filter_gravel_paved(individual_records):
    paved = []
    gravel = []
    not_state_owned = [2, 4]
    for record in individual_records:
        if record['material'] != 7 and record['wearingSurface'] == '1':
            if record['owner'] == 1 and record['adtCategory'] == 'High':
                paved.append(record)
            elif record['owner'] in not_state_owned and record['adtCategory'] == 'Ultra Light':
                gravel.append(record)
            else:
                pass
    return paved, gravel


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
                "deckStructureType":1,
                "material":"$structureTypeMain.kindOfMaterialDesign",
                "wearingSurface":"$wearingSurface/ProtectiveSystem.typeOfWearingSurface",
                "latitude":1,
                "longitude":1,
                "skew":1,
                "lengthOfMaximumSpan":1,
                "bridgeRoadwayWidthCurbToCurb":1,
                "lanesOnStructure":"$lanesOnUnderStructure.lanesOnStructure",
                "designatedInspectionFrequency":1,
            }

    # select states:
    states = ['31'] # Nebraska 

    # years:
    years = [year for year in range(1992, 2020)]
    #years = [year for year in range(1992, 1994)]

    # process precipitation data
    #structBdsMap, structPrecipMap = process_precipitation()

    # process snowfall and freezethaw data
    #structSnowMap, structFreezeMap = process_snowfall()

    # query
    individualRecords = query(fields, states, years, collection)
    #individualRecords = sample_records()
    individualRecords = compute_deck_age(individualRecords)
    print("length after compute deck age: ", len(individualRecords))

    individualRecords = compute_age_1(individualRecords)
    print("length after compute age 1: ", len(individualRecords))

    individualRecords = compute_adt_cat(individualRecords)
    print("length after compute adt cat: ", len(individualRecords))

    paved_ind_rec, gravel_ind_rec = filter_gravel_paved(individualRecords)
    individualRecords =  gravel_ind_rec

    # group records
    groupedrecords = group_records(individualRecords, fields)

    # Integrate baseline difference score, precipitation, freezethaw, and snowfall
    #individualRecords = integrate_ext_dataset_list(structBdsMap,
    #                                               individualRecords,
    #                                               'baseDifferenceScore')
    #individualRecords = integrate_ext_dataset_list(structPrecipMap,
    #                                               individualRecords,
    #                                               'precipitation')
    #individualRecords = integrate_ext_dataset_list(structSnowMap,
    #                                               individualRecords,
    #                                               'snowfall')
    #individualRecords = integrate_ext_dataset_list(structFreezeMap,
    #                                               individualRecords,
    #                                               'freezethaw')

    # Divide grouped records (works only for grouped records)
    groupedrecords = divide_grouped_records(groupedrecords, fields, 2010, 2020)

    # Remove records from specific years (works only for individual records)
    individualRecords = remove_records(individualRecords, 2010, 2019)

    #TODO:
        # Compute intervention from (year: from and to) 
        # Create a for-loop (Although, this loop will be giant)
            # 1. Compute deterioration
            # 2. Intervention
            # 3. Scores
    # change
    groupedrecords = compute_intervention(groupedrecords,
                                          from_to_matrix_kent)

    groupedrecords = compute_intervention(groupedrecords,
                                          from_to_matrix_kent,
                                          component='substructure')
    groupedrecords = compute_intervention(groupedrecords,
                                          from_to_matrix_kent,
                                          component='superstructure')
    #print("\n printing grouped records: ", groupedRecords)

    # Compute deterioration
    groupedRecords = groupedrecords
    groupedRecords = compute_deterioration_slope(groupedRecords,
                                                 component='deck')
    groupedRecords = compute_deterioration_slope(groupedRecords,
                                                 component='substructure')
    groupedRecords = compute_deterioration_slope(groupedRecords,
                                                 component='superstructure')

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

    # Save to the file
    csvFile = 'nebraska_gravel.csv'
    tocsv_list(individualRecords, csvFile)

if __name__=='__main__':
    main()
