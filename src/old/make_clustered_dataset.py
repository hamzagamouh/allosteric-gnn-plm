import json
import os
from tqdm import tqdm

HOME_FOLDER="/storage/praha1/home/hamzagamouh"


from sklearn.model_selection import train_test_split


# datasets=["casbench_sequence_dataset","non_catalytic_csa_sequence_dataset"]

# with open(f"{HOME_FOLDER}/allosteric/method_1/method_1_dataset.fa",'w') as f:
#     for dataset_name in datasets: 
#         dataset=json.load(open(f"{HOME_FOLDER}/allosteric/{dataset_name}.json",'r'))
#         # with open(f"{HOME_FOLDER}/allosteric/method_1/{dataset_name}.fa",'w') as f:
#         for entry,data in tqdm(dataset.items()):
#             sequence=data[0]["sequence"]
#             f.write(f">{entry}\n")
#             f.write(f"{sequence}\n")

        
# ORGANIZE CLUSTERS
    
# all_clusters={}

# with open(f"{HOME_FOLDER}/allosteric/method_1/method_1_clusterRes_cluster.tsv",'r') as f:
#     for line in f:
#         cluster_id,entry=line.strip().split()
#         if cluster_id in all_clusters.keys():
#             all_clusters[cluster_id]+=[entry]
#         else:
#             all_clusters[cluster_id]=[entry]

# json.dump(all_clusters,open(f"{HOME_FOLDER}/allosteric/method_1/method_1_clusters.json",'w'),indent=2)


all_clusters=json.load(open(f"{HOME_FOLDER}/allosteric/method_1/method_1_clusters.json",'r'))

# casbench_clusters={}

# for cluster_id,entries in all_clusters.items():
#     for entry in entries:
#         if "cas" in entry:
#             if cluster_id in casbench_clusters.keys():
#                 casbench_clusters[cluster_id]+=[entry]
#             else:
#                 casbench_clusters[cluster_id]=[entry]


# json.dump(casbench_clusters,open(f"{HOME_FOLDER}/allosteric/method_1/casbench_clusters.json",'w'),indent=2)

# print(len(casbench_clusters),"Casbench clusters")

# all_dataset={}
# for name in ["casbench","non_catalytic_csa"]:
#     dataset=json.load(open(f"{HOME_FOLDER}/allosteric/{name}_sequence_dataset.json",'r'))
#     for entry,vals in dataset.items():
#         all_dataset[entry]=vals
        
# json.dump(all_dataset,open(f"{HOME_FOLDER}/allosteric/sequence_dataset.json",'w'),indent=2)

# all_dataset=json.dump(open(f"{HOME_FOLDER}/allosteric/sequence_dataset.json",'r'))



# # Construct datasets
def make_dataset(dataset_name,all_clusters,X,mode,k,valid_pockets):
    entries=[]
    for cid in X[mode]:
        entries+=[x for x in all_clusters[cid] if x in valid_pockets]
    json.dump(entries,open(f"{HOME_FOLDER}/allosteric/method_2/{dataset_name}_dataset/{mode}_{k}.json",'w'),indent=2)
    return entries





# for dataset_name in ["casbench","non_catalytic_csa"]:
#     all_clusters=json.load(open(f"{HOME_FOLDER}/allosteric/method_1/{dataset_name}_clusters.json",'r'))

#     cluster_ids=list((all_clusters.keys()))
#     for k in range(5):
#         # First split: 80% train, 20% temp (to be split into val/test)
#         X_train, X_temp = train_test_split(
#             cluster_ids, test_size=0.3
#             # ,random_state=42
#         )
#         # Second split: split 20% temp into 10% val and 10% test
#         X_val, X_test = train_test_split(
#             X_temp, test_size=0.5
#             # , random_state=42
#         )
#         X={"train":X_train,"val":X_val,"test":X_test}
#         all_entries={}
#         for mode in X.keys():
#             entries=make_dataset(dataset_name,all_clusters,X,mode,k)
#             all_entries[mode]=entries
#         assert len(set(all_entries["train"]).intersection(set(all_entries["val"])))==0
#         assert len(set(all_entries["train"]).intersection(set(all_entries["test"])))==0
import pandas as pd

df=pd.read_csv(f"{HOME_FOLDER}/allosteric/ahoj_processing/casbench_processed_csa_pockets.csv")
print("Total original pockets",len(df["NON_CATALYTIC_CSA_POCKET"].unique()))
valid_pockets=[x.replace(".pkl","") for x in os.listdir(f"{HOME_FOLDER}/allosteric/method_2/")]
print("Total valid pockets",len(valid_pockets))
all_clusters=json.load(open(f"{HOME_FOLDER}/allosteric/method_1/method_1_clusters.json",'r'))



dataset_name="casbench_gnn"

cluster_ids=[x for x in all_clusters.keys()]
for k in range(5):
    # First split: 80% train, 20% temp (to be split into val/test)
    X_train, X_temp = train_test_split(
        cluster_ids, test_size=0.3
        # ,random_state=42
    )
    # Second split: split 20% temp into 10% val and 10% test
    X_val, X_test = train_test_split(
        X_temp, test_size=0.5
        # , random_state=42
    )
    X={"train":X_train,"val":X_val,"test":X_test}
    all_entries={}
    for mode in X.keys():
        entries=make_dataset(dataset_name,all_clusters,X,mode,k,valid_pockets)
        # all_entries[mode]=[x for x in set(entries).intersection(valid_pockets)]
        all_entries[mode]=entries
    assert len(set(all_entries["train"]).intersection(set(all_entries["val"])))==0
    assert len(set(all_entries["train"]).intersection(set(all_entries["test"])))==0