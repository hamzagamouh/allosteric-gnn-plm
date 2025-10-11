import pandas as pd
import os
import shutil
from tqdm import tqdm
import zipfile

import zipfile
import os
import json

import time
from functools import wraps
import os


AHOJ_DIR="/storage/praha1/home/davidhoksza/projects/ahoj/output-data-latest"
OUTPUT_FILEPATH=f"/storage/praha1/home/hamzagamouh/allosteric/ahoj_processing/ahoj_apo.json"

# TMP_DIR=f"/storage/praha1/home/hamzagamouh/allosteric/ahoj_processing/tmp"
# SCRATCHDIR=os.getenv("SCRATCHDIR")
# TMP_DIR=f"{SCRATCHDIR}/tmp"
TMP_DIR=f"/tmp"

if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)

LOGGING=False



def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper


import tarfile 


def extract_query(query_file):
    query=None
    if query_file.endswith(".tgz"):
        query=query_file.replace('.tgz','')
    if query_file.endswith(".zip"):
        query=query_file.replace('.zip','')

    if LOGGING:
        print("Extracting query : ",query)
    if query_file.endswith(".tgz"):
        # open file 
        file = tarfile.open(f"{AHOJ_DIR}/{query_file}") 
        # extracting file 
        file.extractall(f"{TMP_DIR}/{query}") 
        file.close() 
    if query_file.endswith(".zip"):
        # Specify the directory where you want to extract the contents
        # Create the extraction directory if it doesn't exist
        os.makedirs(f"{TMP_DIR}/{query}", exist_ok=True)

        # Open and extract the zip file
        with zipfile.ZipFile(f"{AHOJ_DIR}/{query_file}", 'r') as zip_ref:
            zip_ref.extractall(f"{TMP_DIR}/{query}")
    
    return query




def remove_query(query_file):
    query=None
    if query_file.endswith(".tgz"):
        query=query_file.replace('.tgz','')
    if query_file.endswith(".zip"):
        query=query_file.replace('.zip','')
    if LOGGING:
        print("Removing query : ",query)

    shutil.rmtree(f"{TMP_DIR}/{query}")
    assert (not os.path.exists(f"{TMP_DIR}/{query}"))

def get_apo_structures(allo_query):
    if LOGGING:
        print(f"Getting apo structures for query {allo_query}")
    apo_file=f"{TMP_DIR}/{allo_query}/apo_filtered_sorted_results.csv"

    if os.path.exists(apo_file):
        df_allo=pd.read_csv(apo_file)
        apo_structures=[x for x in set(df_allo["structure"].tolist())]
    else:
        apo_structures=[]
    
    return apo_structures

def get_holo_structures(query):
    if LOGGING:
        print(f"Getting apo structures for query {query}")
    holo_file=f"{TMP_DIR}/{query}/holo_filtered_sorted_results.csv"
    pdb_id=query.split("-")[0]
    if os.path.exists(holo_file):
        df_allo=pd.read_csv(holo_file)
        holo_structures=[x for x in set(df_allo["structure"].tolist()+[pdb_id])]
    else:
        holo_structures=[pdb_id]
    
    
    return holo_structures

def get_pocket_unps(query):
    if LOGGING:
        print(f"Getting pocket info for query {query}")
    file=f"{TMP_DIR}/{query}/query_pocket_info.csv"
    if os.path.exists(file):
        df_allo=pd.read_csv(file)
        unps=[x for x in set(df_allo["UNPs"].tolist())]
        return unps
    else:
        return None
    
    

def get_ahoj_info(query):
    res_file=f"{TMP_DIR}/{query}/global_results.csv"
    df=None
    if os.path.exists(res_file):
        df=pd.read_csv(res_file)
    
    return df


def get_ahoj_ligands(query,suff=""):
    res_file=f"{TMP_DIR}/{query}{suff}/ligands.json"
    ligs=None
    if os.path.exists(res_file):
        ligs=json.load(open(res_file,'rb'))
    
    return ligs

def get_ahoj_allostery(query):
    res_file=f"{TMP_DIR}/{query}/allostery.json"
    ligs=None
    if os.path.exists(res_file):
        ligs=json.load(open(res_file,'rb'))
    
    return ligs


def get_pocket_residues(query):
    res_file=f"{TMP_DIR}/{query}/pocket_residues.csv"
    df=None
    if os.path.exists(res_file):
        df=pd.read_csv(res_file)
    
    return df

def get_query(query,suff=""):
    if os.path.exists(f"{AHOJ_DIR}/{query}.tgz"):
        query_file=f"{AHOJ_DIR}/{query}.tgz"
    
    elif os.path.exists(f"{AHOJ_DIR}/{query}.zip"):
        query_file=f"{AHOJ_DIR}/{query}.zip"

    else:
        raise FileNotFoundError(f"Query {query} doesn't exist !!")


    if query_file.endswith(".tgz"):
        # open file 
        file = tarfile.open(f"{query_file}") 
        # extracting file 
        file.extractall(f"{TMP_DIR}/{query}{suff}") 
        file.close() 
        assert os.path.exists(f"{TMP_DIR}/{query}{suff}") , f"{TMP_DIR}/{query}{suff} doesn't exist" 
    if query_file.endswith(".zip"):
        # Specify the directory where you want to extract the contents
        # Create the extraction directory if it doesn't exist
        os.makedirs(f"{TMP_DIR}/{query}{suff}", exist_ok=True)

        # Open and extract the zip file
        with zipfile.ZipFile(f"{query_file}", 'r') as zip_ref:
            zip_ref.extractall(f"{TMP_DIR}/{query}{suff}")
    
    return query



def delete_query(query,suff=""):
    shutil.rmtree(f"{TMP_DIR}/{query}{suff}")
    assert (not os.path.exists(f"{TMP_DIR}/{query}{suff}"))

def process_query(query_file):
    allo_query=extract_query(query_file)
    apo_structures=get_apo_structures(allo_query)
    remove_query(query_file)
    return allo_query,apo_structures


if __name__=="__main__":
    # Get all queries
    all_queries = os.listdir(AHOJ_DIR)
    n_all_queries=len(all_queries)
    print("All AHOJ queries",n_all_queries)

    # Get queries by PDB

    with open(f"{OUTPUT_FILEPATH}",'w') as f:
        f.write("{")
        k=0
        for query_file in tqdm(all_queries):
            query,apo_structures=process_query(query_file)
            if len(apo_structures)!=0:
                line=f"'{query}':"+f"{apo_structures}"+","
                f.write(line)
                f.write("\n")
                k+=1

        f.write("}")