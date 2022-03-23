"""
description: functions to process snwofall and freezethaw data

TODO: Snowfall and Freeze thaw
    # Create a function to insert snowfall data into mongodb
    # Create a function to query freezethaw data from mongodb
    # Cheat Code: use precipitation, snowfall, and freeze thaw from already preprocessed.

    Features include:
        1. Creates map: Structure number -> Precipitation
        2. Population and feteching of precipitation data into and from mongodb

    # Ideally the data has to be preprocessed and stores in to the mongodb
    # Write a feature to update the mongodb records

author: Akshay Kale
"""
import csv
from collections import defaultdict
from collections import namedtuple

from maps import *
from nbi_data_chef import *

# What can be stored as a JSON file can also be stored in MONGODB
def read_csv(filename):
    listOfPrecip = list()
    with open(filename, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")
        header = next(csvReader)
        if header[0] ==  '':
            header[0] = 'id'
        header = [col.replace(" ", "") for col in header]
        header = [col.replace("#", "No") for col in header]
        header = [col.replace("(mm)", "") for col in header]
        header = [col.replace(".", "") for col in header]
        header = [col.replace(":", "") for col in header]
        header = [col.replace(".", "") for col in header]
        Record = namedtuple('Record', header)

        for row in tqdm(csvReader, desc='Reading file'):
            record = Record(*row)
            listOfPrecip.append(record)
    return listOfPrecip, header

def reverse_dict(dictionary):
    """
    Description:
    """
    newDict = defaultdict()
    for key, value in zip(dictionary.keys(), dictionary.values()):
        newDict[value] = key
    return newDict

def make_fips_precp(listOfPrecip, state_code_mapping):
    """
    Description:
    """
    #TODO: Incomplete functions
    # query mongodb
    fips_precp_dict = defaultdict()
    #for record in listOfPrecip:
    return fips_precp_dict

def query_db():
    # define query
    fields ={"_id":0,
             "structureNumber":1,
             "countyCode":1,
             "stateCode":1
            }

    years = [2016]
    #states = code_state_mapping.keys()
    states=[31]
    nbiDB = get_db()
    collection = nbiDB['nbi']
    #print(collection)
    results = query(fields, years, states, collection)
    #print(results)
    return results


def generate_map(listOfRecords, value, key='structurenumber'):
    """
    description:
       generates a map of structure number and the selected value
    args:
        listofrecords (list of named tuple):
        value (string):
        key (string):
    returns:
        generatedmap (dictionary):
    """
    generatedMap = defaultdict()
    for record in listOfRecords:
        fieldValue = record._asdict()[value]
        keyValue = record._asdict()[key]
        keyValue = keyValue[:-3].strip(" ")
        generatedMap[keyValue] = fieldValue
    return generatedMap

def process_snowfall():
    #filename = 'precipitation-allstates-2011.csv'
    #state_code_mapping = reverse_dict(code_state_mapping)
    #listOfPrecip, header= read_csv(filename)
    #records = query_db()

    #### process data through a cheat file ####
    cheatFilename = 'decision_tree.csv'
    listOfSnowfallFreeze, cheatHeader = read_csv(cheatFilename)
    structSnowMap = generate_map(listOfSnowfallFreeze, 'snowfall')
    structFreezeMap = generate_map(listOfSnowfallFreeze, 'freezethaw')
    return structSnowMap, structFreezeMap

if __name__ =='__main__':
    process_snowfall()

