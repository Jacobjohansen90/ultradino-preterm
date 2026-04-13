#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
@author: jj@di.ku.dk

A full implementation of this template can be found here:
https://github.com/Jacobjohansen90/ultradino-preterm/tree/main/preprocessing
in the preprocess.py and workers.py scripts    
"""

#%% Imports
import sqlite3
import csv
import json 
import multiprocessing as mp
import logging
import time

#%% Sqlite Database
"""
In order to get an overview of the database:
Open a terminal where the database is located and run the following commands
    >>> sqlite3
    >>> .open DATABASENAME.sqlite
    >>> .schema
This should print an overview of the tables in the database.
Exit the database using .quit
"""



#%% Variables

#Number of threads for MP
num_workers = 60

#Path to CSV containing phair_cpr_hashes and sqlite database
path_to_EHR_csv = "/some/path/csv.csv"
path_to_US_db = "/some/path/db.sqlite"
save_path = "/some/path/dict.json"

#CSV indexes we want in the final output
variables_from_ehr = ['GA_days',
                      'cpr_child',
                      'cpr_mother',
                      'Birthdate',
                      'Hospital',]

#Sqlite database indexes we want in the final output
variables_from_db = ['file_path',
                     'manufacturer',
                     'manufacturer_model',
                     'study_date',
                     'physical_delta_x',
                     'physical_delta_y']


#%%Loop the CSV and DB to find the index of the variables
ehr_idx = {}
db_idx = {}

f = open(path_to_EHR_csv)
EHR_csv = csv.reader(f)

ehr_headers = next(EHR_csv)

#We need to get the total amount of births
births = sum(1 for line in EHR_csv)


for i in range(len(ehr_headers)):
    for variable in variables_from_ehr:
        if ehr_headers[i] == variable:
            ehr_idx[variable] = i

if len(variables_from_ehr) != len(ehr_idx):
    found = list(ehr_idx.keys())
    diff = list(set(variables_from_ehr) - set(found))
    raise Exception(f"Did not find variables {diff} in EHR csv")

#Crawl DB for variables indexes
with sqlite3.connect(path_to_US_db) as con:
    cur = con.cursor()
    cur.execute("SELECT * FROM metadata_cache LIMIT 0")
    db_headers = [desc[0] for desc in cur.description]

for i in range(len(db_headers)):
    for variable in variables_from_db:
        if db_headers[i] == variable:
            db_idx[variable] = i

if len(variables_from_db) != len(db_idx):
    found = list(db_idx.keys())
    diff = list(set(variables_from_db) - set(found))
    raise Exception(f"Did not find variables {diff} in DB")

f.close()

#%% Define worker functions

def csv_extracter(path_to_csv, csv_que, done):
    """
    This function loads the CSV info, including the phair_cpr_hash

    Parameters
    ----------
    path_to_csv : str
        path to csv
    csv_que : mp.Queue()
        mp.Queue where we put the extracted csv rows
    done : mp.Value
        shared memory across processes telling the crawlers the csv_extractor is done
    """
    f = open(path_to_csv)
    f_csv = csv.reader(f)
    #Load headers and throw them away
    _ = next(f_csv)
        
    for row in f_csv:
        csv_que.put(row)
        #Avoid flooding the queue. Not strictly necessary, but preserve memory
        if csv_que.qsize() > 5000:
            time.sleep(1)            
    #Set the shared value true, so the crawlers know no more csv rows are comming
    done.value = True
    f.close()

def db_crawler(ehr_idx, db_idx, path_to_db, csv_que, data_que, done):
    """
    Parameters
    ----------
    csv_idx : Some structure holding the indexes we want from the CSV.
        Indexes used from csv
        This functionality could also be done in the csv_extractor function
    db_idx : Some structure holding the indexes we want from the database.
        Indexes used from sqlite database
    path_to_db : str
        path to the sqlite database
    csv_que : mp.Queue()
        mp.Queue where we get the extracted csv rows
    data_que : mp.Queue()
        mp.Queue where we put the processed data
    done : mp.Value
        shared memory across processes telling the crawlers the csv_extractor is done
    """
    #Connect to database and set cursor
    con = sqlite3.connect(path_to_db)
    cur = con.cursor()
    #Loop
    while not done.value or csv_que.qsize() > 0:
        #Get row, extract cpr_phair_hash and link to database
        row = csv_que.get(timeout=30)
        cpr_phair_mother = row[ehr_idx['cpr_phair_mother']]

        #This links the phair_hash to the filehash in the database
        query = f"SELECT xxhash FROM cpr_hashes WHERE phair_hash = '{cpr_phair_mother}'"
        cpr_hashes = list(cur.execute(query))


        if len(cpr_hashes) == 0:
            """
            Some phair_hashes do not have an entry in the database.
            This is likely caused by the ultrasound not being in the ultrasound dataset
            Deal with this situation appropiately
            """
            continue
        
        else:    
            #Here we gather the data into a dictionary. 
            #Use whatever structure is appropriate.
            data_temp = {}
  
            #Put the csv variables we want into the dictionary
            for key in ehr_idx.keys():
                data_temp[key] = row[ehr_idx[key]]
    
            #We now get the images associated with the phair_cpr_hash
            imgs = []
            
            #Each mother can have multiple cpr_hashes. 
            #This normally corresponds to different pregnancies
            #Make sure to filter the pregnancies you want using the study date and birthdate
            for cpr_ in cpr_hashes:
                cpr = cpr_[0]
                try:
                    query = f"SELECT * FROM metadata_cache WHERE file_hash = '{cpr}'"
                    entries = list(cur.execute(query))
                except:
                    """
                    In a few cases the cpr might be corrupted and result in an UTF-8 encoding error.
                    Deal with this situations appropiately
                    """
                    continue
    
                if len(entries) == 0:
                    """
                    The database may have a placeholder for the cpr, but no actual data
                    Deal with this situations appropiately
                    """
                    continue
                else:    
                    for entry in entries:
                        img_temp = {}
                        if entry[db_idx['img_path']] is None:
                            """
                            This means the image exists on the SONAI server but is not on NGC
                            The SONAI path can be extracted as entry[0]
                            Deal with this situations appropiately
                            """
                            continue
                        else:
                            for key in db_idx.keys():
                                img_temp[key] = entry[db_idx[key]]
                        imgs.append(img_temp)

            if len(imgs) > 0:
                data_temp['imgs'] = imgs
                data_que.put(data_temp)
            else:
                """
                We found no images for this phair entry, even though there was a database entry.
                Deal with this situations appropiately
                """
                continue

#%% Setup ques, loggers and start processes

csv_que = mp.Queue()
data_que = mp.Queue()
done = mp.Value('b', False)
csv_size = mp.Value('i', births) #This value is calculated in line 63 

logging.basicConfig(filename="/some/path/log.log", filemode='w')
logger = logging.getLogger("Your Preprocessor")
logger.setLevel(logging.INFO)

processes = []
p = mp.Process(target=csv_extracter, args=(path_to_EHR_csv, csv_que, done))
p.start()
processes.append(p)

#Check that we have the workers we want avaliable. Leave a little CPU overhead
num_workers = min(num_workers, mp.cpu_count()-4)

logger.info(f"Starting {num_workers} workers")

for i in range(num_workers):
    p = mp.Process(target=db_crawler, args=(ehr_idx, db_idx, path_to_US_db, csv_que, data_que, done))
    p.start()
    processes.append(p)

#%% Crawl the data_que and save results

#Here we just dump the outputs into a JSON. 
#You might want to do something else / more

final_data = {}
counter = 1

#Here we use a dummy index - counter
#Be aware that the mothers cpr phair hash might have multiple entries and result in collisions
#Using the childs cpr phair hash is a safe option with no collisions

while counter < csv_size.value:
    data = data_que.get()
    final_data[str(counter)] = data
    counter += 1
    if counter % 10000 == 0:
        logger.info(f"Processed {counter} cprs")
    
with open(save_path, 'w') as file:
    json.dump(final_data, file)
    
#Shut down the processes.
for p in processes:
    p.join()
    
