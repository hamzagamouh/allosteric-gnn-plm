import requests
import json
from tqdm import tqdm
import os

HOME_FOLDER="/storage/praha1/home/hamzagamouh"
PDB_FOLDER=f"{HOME_FOLDER}/allosteric/pdb_files"
url = 'https://passer.smu.edu/api'

dataset=json.load(open(f"{HOME_FOLDER}/allosteric/sequence_dataset.json",'r'))

# def get_fpocket(pocket):
#     data=dataset[pocket]
#     pdb_id=data[0]["pdb_id"].upper()
#     chain_id=data[0]["chain_id"].upper()
#     data = {"pdb": pdb_id, "chain": chain_id}
#     results = requests.post(url, data=data).content
#     with open(f"{HOME_FOLDER}/allosteric/method_2/fpocket_feats/{pocket}.zip","wb") as f:
#         f.write(results)



def get_fpocket(pocket):
    data=dataset[pocket]
    pdb_id=data[0]["pdb_id"].lower()
    chain_id=data[0]["chain_id"].upper()
    print(f"{pocket}_{pdb_id}_{chain_id}")
    # os.system(f"fpocket -f {PDB_FOLDER}/{pdb_id}.pdb -k {chain_id}")
    # os.system(f"mv .{PDB_FOLDER}/{pdb_id}_out {HOME_FOLDER}/allosteric/method_2/fpocket_feats/{pocket}")



# from joblib import Parallel, delayed
# Parallel(n_jobs=-1,backend="multiprocessing",verbose=1)(delayed(get_fpocket)(pocket) for pocket in dataset.keys())
fpocket_mapping=json.load(open(f"{HOME_FOLDER}/allosteric/fpocket_mapping.json",'r'))

processed=[x for x in fpocket_mapping.keys()]+[x for x in dataset.keys() if "cas" in x]

for pocket in dataset.keys():
    if pocket not in processed:
        get_fpocket(pocket)