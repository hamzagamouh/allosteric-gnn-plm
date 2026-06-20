import os
import json
HOME_FOLDER="/storage/praha1/home/hamzagamouh"
all_dataset={}
for name in ["casbench","non_catalytic_csa"]:
    dataset=json.load(open(f"{HOME_FOLDER}/allosteric/{name}_sequence_dataset.json",'r'))
    for entry,vals in dataset.items():
        all_dataset[entry]=vals
        
json.dump(all_dataset,open(f"{HOME_FOLDER}/allosteric/sequence_dataset.json",'w'),indent=2)