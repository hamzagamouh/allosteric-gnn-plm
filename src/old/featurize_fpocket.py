import pandas as pd
import os
import traceback


from Bio.PDB import *

PDB_PARSER = PDBParser()

HOME_FOLDER="/storage/praha1/home/hamzagamouh"

FPOCKET_FOLDER=f"{HOME_FOLDER}/allosteric/method_2/fpocket_feats"


from Bio.PDB.Polypeptide import one_to_three,three_to_one,is_aa

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from extract_ahoj_apo import *
import numpy as np

import json
non_catalytic_pockets=json.load(open(f"{HOME_FOLDER}/allosteric/ahoj_processing/non_catalytic_csa_pockets_info.json",'r'))
non_catalytic_pockets_queries=json.load(open(f"{HOME_FOLDER}/allosteric/ahoj_processing/non_catalytic_csa_pockets.json",'r'))
dataset=json.load(open(f"{HOME_FOLDER}/allosteric/sequence_dataset.json",'r'))


def get_best_fpocket(query,pocket,chain_id,query_res):
    fpocket_results=f"{FPOCKET_FOLDER}/{pocket}/pockets"
    best_p_hits=0
    best_pocket=""
    for file in os.listdir(fpocket_results):   
        if not file.endswith(".pdb"):
            continue
        fpocket_name=file.replace("_atm.pdb","")
        structure = PDB_PARSER.get_structure("X",f"{fpocket_results}/{file}")
        # Loop through all chains in the structure
        for model in structure:
            for chain in model:
                fpocket_res=[]
                if chain_id==str(chain.get_id()):
                    # Search for residue by name and position
                    for residue in chain:
                        # if is_aa(residue):
                        name=residue.get_resname().upper()
                        res_id=f"{chain_id}_{residue.get_id()[1]}"
                        fpocket_res+=[res_id]
                # if pocket=="pocket_5":
                #     print(query,pocket,chain_id,str(chain.get_id()))
                #     print("fpocket_res",set(fpocket_res))
                #     print("query_res",set(query_res))
                p_hits=len(set(fpocket_res).intersection(set(query_res)))*100/len(set(query_res))
                if p_hits>best_p_hits:
                    best_p_hits=p_hits
                    best_pocket=fpocket_name
    if best_p_hits>0:
        # print(f"Query {query} - Pocket : {pocket} --> Detected pocket {best_pocket} with {best_p_hits}% match ")
        return best_pocket
    else:
        # print(f"Query {query} - Pocket : {pocket} --> No matching pocket found !")
        pass
                






def get_valid_query(pocket):
    valid_queries=[]
    queries=non_catalytic_pockets_queries[pocket]
    assert os.path.exists(f"{FPOCKET_FOLDER}/{pocket}") , f"{pocket} doesn't exist"
    for q in queries:
        pdb_id=q.split("-")[0]
        if os.path.exists(f"{FPOCKET_FOLDER}/{pocket}/{pdb_id}_out.pdb"):
            valid_queries+=[q]
    return valid_queries
        
def extract_fpocket_feats(pocket):
    fpocket_result_dir=f"{FPOCKET_FOLDER}/{pocket}"
    all_pockets_feats={}
    for file in os.listdir(fpocket_result_dir):
        if file.endswith("info.txt"):
            with open(f"{fpocket_result_dir}/{file}",'r') as f:
                for line in f:
                    if line.startswith("Pocket"):
                        pocket_name=line.strip().lower().replace(" ","").replace(":","")
                        all_pockets_feats[pocket_name]=[]
                    else:
                        try:
                            feat_name,feat_val=line.strip().split(":")
                            feat_val=float(feat_val.strip())
                            all_pockets_feats[pocket_name]+=[feat_val]
                        except:
                            continue

    return all_pockets_feats


def match_fpocket():
    fpocket_mapping={}
    count=0

    for pocket,info in tqdm(non_catalytic_pockets.items()):
        try:
            valid_queries=get_valid_query(pocket) 
            # print(valid_queries)
            for query in valid_queries:
                get_query(query)
                pdb_id,_,_,_=query.split("-")
                pocket_info=get_pocket_residues(query)
                pocket_res=pocket_info["mapped_binding_residues"].iloc[0].split()
                query_residues={}
                for x in pocket_res:
                    cid,pos=x.split("_")
                    if cid not in query_residues.keys():
                        query_residues[cid]=[x]
                    else:
                        query_residues[cid]+=[x]
                # query_res=pocket_res
                for chain_id,query_res in query_residues.items():
                    query_res=query_residues[chain_id]
                    best_fpocket=get_best_fpocket(query,pocket,chain_id,query_res)
                    if best_fpocket:
                        fpocket_mapping[pocket]=best_fpocket
                        pocket_feats=extract_fpocket_feats(pocket)[best_fpocket]
                        np.save(f"{HOME_FOLDER}/allosteric/method_2/fpocket_feats/{pocket}_fp_feats.npy",np.array(pocket_feats))
                        count+=1
                        # break
                delete_query(query)
            if fpocket_mapping.get(pocket,"")=="":
                print("No Fpocket pocket found for pocket",pocket)
        except Exception as e:
            print(f"skipping pocket {pocket} due to error \n {e}")
            # traceback.print_exc()


    import json
    json.dump(fpocket_mapping,open(f"{HOME_FOLDER}/allosteric/fpocket_mapping.json",'w'),indent=4)


match_fpocket()


